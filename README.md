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

## SIMCALC GPU jobs

Upload the project scripts and the selected system's input bundle to SIMCALC,
and install the project's NVIDIA environment in its shared `.venv` first. From
the project directory on the SIMCALC login node, check a new replica with:

```bash
bash simcalc/main.sh --check-only abeta42_7q4m_8bfa_replica_2
```

Then submit it by leaving out `--check-only`:

```bash
bash simcalc/main.sh abeta42_7q4m_8bfa_replica_2
```

The system name selects matching input and output directories under
`data/inputs/openmm_native/solution/` and `data/outputs/openmm_native/solution/`.
The launcher refuses existing output, execution locks, and an already queued
or running job for that system. It starts fresh runs only; it never deletes
results or stale locks, resumes a checkpoint, or installs packages automatically.
If a job is interrupted, inspect its log and checkpoint before recovering it.

Before submission, the launcher checks files, writable destinations, Slurm
connectivity, and resource capacity. `--check-only` runs these checks and asks
Slurm for an estimate without submitting a job; it can create parent/log
directories and temporary write probes. Reported filesystem space does not
establish your account's quota or guarantee enough space for the whole run.

Slurm chooses a GPU node with one GPU, four CPU cores, 16 GB of system RAM, and
up to seven days of elapsed time. Busy nodes remain eligible: the job queues
normally, and an unavailable start estimate is not a submission failure. Nodes
marked down, drained, failed, unknown, or under maintenance are excluded from
the early capacity check. The job continues after you disconnect. Slurm stores
the submitted batch script, but jobs read helpers and Python code from the
shared project directory: avoid editing those files while jobs are queued or
running. Previously submitted standalone scripts, such as job 203, do not use
the new helpers.

Once allocated, the job checks glibc 2.34 or newer, the shared Python environment,
and visibility of the allocated GPU. The existing simulation workflow validates
the input bundle and initializes CUDA for each real stage before creating its
output directory; there is no separate test simulation. CUDA is mandatory; CPU
cores still handle preparation and coordination. Logs are saved as
`simcalc/logs/SYSTEM_ID-JOB_ID.log`. The launcher
prints the job ID and commands for checking its status and reading its log.

### Submit from PyCharm

Upload the updated `simcalc` folder through your existing Deployment connection.
Under **Settings → Tools → Remote SSH External Tools**, add or update the tool:

| Field | Value |
| --- | --- |
| Program | `/bin/bash` |
| Arguments | `/nfs/carlos/on-boarding-phospholipids-openmm/simcalc/main.sh abeta42_7q4m_8bfa_replica_2` |
| Working directory | `/nfs/carlos/on-boarding-phospholipids-openmm` |
| SSH configuration | Your existing SIMCALC connection |

Enable **Open console for tool output**. Run it from **Tools → Remote Tools**;
SSH connection or authentication failures appear in PyCharm. To check without
submitting, put `--check-only` before the system name in Arguments. For another
replica, upload its input bundle and change the system name.

## Runtime Notes

- Every stage writes `checkpoint.chk` in its output directory at the same
  interval as trajectory frames. OpenMM replaces this file atomically, so a
  crash can lose at most one trajectory-reporting interval of computation.
- Rerun the same single-step command after a crash to load the checkpoint,
  append to `state_data.csv` and `trajectory.dcd`, and run only the remaining
  steps. Rerunning a sweep also skips completed stages before resuming the
  interrupted stage.
- A checkpoint is removed only after `final_state.xml` and
  `final_coordinates.pdb` are written successfully. If checkpoint loading
  fails, the run stops without overwriting partial outputs.
- Binary checkpoints require the same system definition, OpenMM version,
  platform, and hardware. They are intended for continuing on the same PC;
  completed stages continue to use the more portable `final_state.xml`.
- On Linux AMD hosts with ROCm installed, the package preloads the HIP runtime
  before OpenMM import so `HIP` can be used when available.
- On NVIDIA hosts, normal OpenMM `CUDA` selection remains unchanged.
- On CPU-only hosts, the platform fallback order still reaches `CPU`.
- Use `OPENMM_DEVICE_INDEX` to pin a specific GPU when multiple devices are
  visible.
- Use `CHARMM_GUI_MD_OPENMM_SKIP_BOOTSTRAP=1` to skip runtime bootstrap.
