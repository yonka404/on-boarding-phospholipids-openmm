# membrane-openmm

Minimal Python/OpenMM scaffold to solve the membrane-size exercise with a reproducible workflow.

This repo is designed for systems prepared with **CHARMM-GUI Membrane Builder** and then run with **OpenMM**.
It reads the CHARMM-style `psf`/`pdb`/`toppar` files directly, so you do not need to launch NAMD stage-by-stage by hand.

## Folder layout

```text
membrane-openmm/
├── pyproject.toml
├── README.md
├── systems/
│   ├── n50/
│   │   └── charmm-gui/
│   │       ├── step5_assembly.psf
│   │       ├── step5_assembly.pdb
│   │       ├── step5_assembly.str
│   │       ├── toppar.str
│   │       └── toppar/
│   ├── n100/
│   │   └── charmm-gui/
│   └── n150/
│       └── charmm-gui/
├── results/
├── scripts/
│   ├── run_case.py
│   ├── run_sweep.py
│   └── analyze_size.py
└── src/
    └── membrane_openmm/
        ├── __init__.py
        ├── charmm_gui.py
        └── pipeline.py
```

## What this scaffold does

- loads a membrane directly from CHARMM-GUI files
- runs staged equilibration + production from one Python entry point
- writes per-stage trajectory, checkpoint, state, final PDB, and CSV logs
- computes box area, area-per-lipid, thickness proxy (`Lz`) and basic thermodynamics
- makes it easy to compare `N=50`, `N=100`, `N=150`

## Important limitation

This scaffold mirrors the **idea** of your NAMD workflow (staged equilibration followed by production), but it does *
*not** attempt a byte-for-byte port of CHARMM-GUI's NAMD-specific colvars/extra-bond restraint files. For the course
exercise this is usually fine, because the goal is to compare size trends consistently across systems.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m openmm.testInstallation
```

## Prepare your systems

Generate three membrane systems with CHARMM-GUI, for example:

- `systems/n50/charmm-gui/`
- `systems/n100/charmm-gui/`
- `systems/n150/charmm-gui/`

Each `charmm-gui/` folder should contain at least:

- `step5_assembly.psf`
- `step5_assembly.pdb`
- `step5_assembly.str`
- `toppar.str`
- `toppar/`

## Run one case

```bash
python mains/run_case.py \
  --inputs-root inputs/n100 \
  --outdir results/n100 \
  --platform CPU \
  --temperature 303.15
```

If you have a supported GPU, replace `CPU` with `HIP`, `CUDA`, or `OpenCL`.

## Run the full size sweep

```bash
python mains/run_sweep.py \
  --inputs inputs/n50/charmm-gui inputs/n100/charmm-gui inputs/n150/charmm-gui \
  --results-root results \
  --platform CPU \
  --temperature 303.15
```

## Analyze the size effect

```bash
python mains/analyze_size.py --results-root results
```

This creates a CSV summary at:

```text
results/size_summary.csv
```

## Suggested discussion points for exercise 2

Once the runs finish, compare across sizes:

- average area per lipid
- average `Lz` (box height proxy)
- density / potential energy stabilization
- whether the smallest system fluctuates more strongly
- whether larger systems look smoother / less noisy but cost more time

