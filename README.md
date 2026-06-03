# CHARMM-GUI MD

This repository runs CHARMM-GUI-generated systems with Python and OpenMM. It
supports a single solute in either a membrane or a water-and-salts solution.
Only CHARMM-GUI's OpenMM-native export format is supported.

The public APIs are environment-specific:

- `charmm_gui_md.membrane`: uses `step5_input`, then `step6.1_equilibration`
  through `step7_production`.
- `charmm_gui_md.solution`: uses `step3_input`, then `step4_equilibration` and
  `step5_production`.

Reusable input validation, protocol parsing, simulation setup, restraints,
restart handling, and reporters live under `charmm_gui_md.shared`. Protocol
settings decide whether OpenMM uses a membrane or isotropic barostat.

## Data Layout

OpenMM-native inputs and outputs are grouped by system kind and system ID:

```text
data/
  inputs/
    openmm_native/
      membrane/
        ligand_membrane/
      solution/
        abeta_40/
  outputs/
    openmm_native/
      membrane/
        ligand_membrane/
      solution/
        abeta_40/
```

Each OpenMM-native system directory contains an `openmm/` folder and any
sibling parameter directories referenced by `openmm/toppar.str`. Outputs mirror
the input format, system kind, and system ID so results remain traceable.

The adapter requires assignment-style OpenMM `.inp` files, JSON
`openmm/sysinfo.dat`, and one parameter file path per `openmm/toppar.str` line.
Raw CHARMM input commands are not supported.

For the `abeta_40` CHARMM-GUI Solution Builder export, the runtime bundle is:

```text
openmm/step3_input.psf
openmm/step3_input.pdb
openmm/step3_input.crd
openmm/step4_equilibration.inp
openmm/step5_production.inp
openmm/sysinfo.dat
openmm/toppar.str
openmm/restraints/prot_pos.txt
toppar/  # the 56 files referenced by openmm/toppar.str
```

Do not include Solution Builder construction files, logs, OpenMM helper scripts,
`openmm/README`, or the unreferenced `toppar/tip216.crd` in the runtime bundle.
The OpenMM-native adapter validates referenced parameter files and restraint
files required by active stage protocols; it does not require a `lig/` folder.

## Running

Scripts select a system by its directory identifier and derive mirrored input
and output paths under `data/`. Single-step commands also require a valid stage
name.

```bash
uv run python mains/run_membrane_single_step.py ligand_membrane step6.1_equilibration
uv run python mains/run_membrane_sweep.py ligand_membrane
```

```bash
uv run python mains/run_solution_single_step.py abeta_40 step4_equilibration
uv run python mains/run_solution_sweep.py abeta_40
```

## Runtime Notes

- On Linux AMD hosts with ROCm installed, the package preloads the HIP runtime
  before OpenMM import so `HIP` can be used when available.
- On NVIDIA hosts, normal OpenMM `CUDA` selection remains unchanged.
- On CPU-only hosts, the platform fallback order still reaches `CPU`.
- Use `OPENMM_DEVICE_INDEX` to pin a specific GPU when multiple devices are
  visible.
- Use `CHARMM_GUI_MD_OPENMM_SKIP_BOOTSTRAP=1` to skip runtime bootstrap.
