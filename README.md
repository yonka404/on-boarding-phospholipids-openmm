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
the project directory on the SIMCALC login node, submit a fresh replica with:

```bash
python3 simcalc/main.py SYSTEM_ID
```

Replace `SYSTEM_ID` with the exact input system/replica folder name.
**Every run automatically performs the prerequisite checks, then submits one
job if they pass.** There is no separate checking mode or flag to remember.
Any failed prerequisite stops the launcher with an error before submission.

The system name selects matching input and output directories under
`data/inputs/openmm_native/solution/` and `data/outputs/openmm_native/solution/`.
The launcher refuses existing output, execution locks, and an already queued
or running job for that system. It starts fresh runs only; it never deletes
results or stale locks, resumes a checkpoint, or installs packages automatically.
If a job is interrupted, inspect its log and checkpoint before recovering it.

Before submission, the launcher checks files, writable destinations, Slurm
connectivity, duplicate jobs, partition limits, and resource capacity. It can
create parent/log directories and temporary write probes during these checks.
Reported filesystem space does not establish your account's quota or guarantee
enough space for the whole run. Slurm's start-time prediction is advisory: a
missing estimate produces a warning, then actual submission determines acceptance.

Slurm chooses a GPU node with one GPU, four CPU cores, 16 GB of system RAM, and
up to seven days of elapsed time. Busy nodes remain eligible: the job queues
normally, and an unavailable start estimate is not a submission failure. Nodes
marked down, drained, failed, unknown, or under maintenance are excluded from
the early capacity check. The job continues after you disconnect. Slurm stores
the submitted batch script, but jobs read helpers and Python code from the
shared project directory: avoid editing those files while jobs are queued or
running.

The submission launcher and node runner use only Python's standard library and
support system Python 3.8 or newer. They do not import OpenMM. The node runner
checks glibc 2.34 or newer before starting the shared `.venv/bin/python`, which
must meet the simulation project's Python requirement (3.12.10 or newer).
Do not use the project environment or `uv run` to launch submission on the older
login node.

GPU-node checks happen automatically **after Slurm allocates a node**: files and
output protection are rechecked, then the environment and allocated GPU are
validated. The existing simulation workflow validates the input bundle and
initializes CUDA for each real stage before creating its output directory;
there is no separate test simulation. A GPU initialization failure stops the job
before that stage writes output. CUDA is mandatory; CPU cores still handle
preparation and coordination.

Logs are saved as `simcalc/logs/SYSTEM_ID-JOB_ID.log`. The launcher prints the job
ID and commands for checking its status and reading its log. Owned locks are
released on normal completion or failure; interruption stops the active child
before releasing its lock. A forced kill or node crash can leave a lock for
manual inspection. If submission is interrupted or returns an unrecognized
reply, check `squeue` before retrying: Slurm might already have accepted it.

The code is split into `simcalc/main.py` (submission), `simcalc/common.py`
(shared checks and process handling), and `simcalc/run_solution.py` (allocated
job). `simcalc/validate_solution.py` reports the project environment and GPU
allocation. The small `simcalc/run_solution.slurm` wrapper holds resource
requests, loads the CUDA module, and starts the Python node runner.

### Submit from PyCharm

These instructions follow PyCharm Pro 2026.2. Deployment uploads the code; a
Remote SSH External Tool runs the submission launcher on the SIMCALC login node.

1. Keep the existing **SIMCALC** SFTP Deployment configuration under
   **Settings → Build, Execution, Deployment → Deployment**. Its connection uses
   `carlos@simcalc.upc.es`, with root path `/nfs/carlos`. Under **Mappings**, the
   local project folder maps to deployment path `/on-boarding-phospholipids-openmm`.
   Together these select `/nfs/carlos/on-boarding-phospholipids-openmm` remotely.
2. Before uploading, keep `data`, `.venv`, `.git`, `.idea`, and `simcalc/logs`
   excluded from routine code uploads. Git ignore rules alone do not configure
   Deployment exclusions.
   Transfer the selected system's input bundle separately; missing inputs will
   stop submission automatically.
3. In the Project panel, right-click the project folder and choose
   **Deployment → Upload to SIMCALC** (or **Upload to… → SIMCALC**). Wait for the
   upload to finish in **File Transfer**. If you already use automatic Deployment
   uploads, still wait for transfers to finish before submitting.
4. Open **Settings → Tools → Remote SSH External Tools** and add or update a tool
   with these fields. All paths below are **remote** paths:

| Field | Value |
| --- | --- |
| Name | `Submit SIMCALC solution` |
| Group | `Remote Tools` |
| Program | `/usr/bin/python3` |
| Arguments | `-u /nfs/carlos/on-boarding-phospholipids-openmm/simcalc/main.py SYSTEM_ID` |
| Working directory | `/nfs/carlos/on-boarding-phospholipids-openmm` |
| SSH Configuration | The existing connection for `carlos@simcalc.upc.es` |

Replace `SYSTEM_ID` in Arguments with a fresh replica's folder name. Under
**Advanced Options**, enable **Open console for tool output**. The option
**Synchronize files after execution** is not a code upload before submission;
leave it disabled for this tool and use Deployment for uploads.

Run **Tools → Remote Tools → Submit SIMCALC solution**. The console shows all
submission checks and, on success, the job ID. An SSH authentication failure
appears in PyCharm before the launcher runs. The tool finishing means submission
finished; use the printed monitoring commands to follow the queued/running job.
For another replica, upload its inputs and change the system ID in Arguments.

If **Remote SSH External Tools** is missing, check that the **FTP/SFTP/WebDAV
Connectivity** and **HTTP Client** plugins are enabled. See JetBrains'
[remote tool setup](https://www.jetbrains.com/help/pycharm/configuring-third-party-tools.html),
[tool settings](https://www.jetbrains.com/help/pycharm/settings-tools-remote-ssh-external-tools.html),
and [Deployment upload instructions](https://www.jetbrains.com/help/pycharm/uploading-and-downloading-files.html).

When upgrading from the Bash launcher, update any saved tool commands to the
Python command above. Deployment can leave deleted local files on the server:
obsolete `simcalc/main.sh` and `simcalc/common.sh` may still be present there.
Remove them manually only after no queued/running jobs depend on them. Do not
enable broad remote deletion to perform this cleanup.

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
