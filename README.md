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

Start in `mains/`. There are two public launchers:

| Launcher | Where it runs | Supported runs |
| --- | --- | --- |
| `mains/local.py` | Your PC | Membrane or solution; full protocol or one stage |
| `mains/simcalc.py` | SIMCALC login node | Start or recover a full solution protocol through Slurm |

Run these examples from the repository root:

```bash
# Full local protocols
uv run python mains/local.py solution abeta_40
uv run python mains/local.py membrane ligand_membrane

# One local stage
uv run python mains/local.py solution abeta_40 --stage step4_equilibration
uv run python mains/local.py membrane ligand_membrane --stage step6.1_equilibration

# Submit from the SIMCALC login node using system Python
python3 mains/simcalc.py SYSTEM_ID
```

Omitting `--stage` runs the full protocol. Stage names are validated against the
selected system type; an invalid stage lists its allowed names. Both launchers
accept `--help`.

The system ID selects mirrored input and output directories under `data/`.
Paths are resolved from the repository location, independently of the working
directory. In an IDE or another directory, use the absolute script path and,
for local runs, the project's `.venv` interpreter.

Launch support lives in `mains/_common.py` and `mains/_simcalc/`; the latter's
scripts are internal Slurm helpers. Shared console formatting lives in
`mains/_console.py`, and read-only status collection and rendering in
`mains/_status.py`. Simulation implementations and public workflow APIs live
in `src/charmm_gui_md/`.

## Checking simulation status

Use the same command on your local PC or the SIMCALC login node:

```bash
python3 mains/status.py
```

It prints one snapshot and exits. Run it again for an update, or use another
terminal while a local simulation runs. It needs only system Python 3.8+;
OpenMM, the project environment, and Slurm are not required on your PC.
It reads this checkout's results and activity on the machine where it runs;
it does not connect from your PC to SIMCALC.

The report includes solution and membrane systems with outputs, existing
SIMCALC job logs, or a matching active launcher/job. Systems with inputs alone
are excluded. Active systems come first, then unfinished or unknown systems,
then completed systems. Completed systems are included automatically.

An illustrative comparison table is:

```text
System                                Job/PID  State    Stage              Stage %  ns/day  Stage ETA  Report age
solution/abeta42_7q4m_8bfa_replica_2  321      RUNNING  2/2 Production       64.0%    13.4  ~13d 10h   28s
solution/abeta42_7q4m_8bfa_replica_3  322      QUEUED   1/2 Equilibration        -       -  -          -
```

Full system names are preserved. The stage number is its position in the full
protocol; `2/2 Production` means the second of two stages. Local processes use
identifiers such as `PID 4128`. Short details follow the table for active and
unfinished systems: the exact stage name, simulated/target nanoseconds and
steps, completed-stage count, and available job timing, node, and log path.
Queued jobs show their waiting reason. Missing or unreadable information is
explained there. Narrow terminals automatically use stacked blocks with the
same information; there are no extra monitoring flags or detail modes.

`RUNNING` confirms a live launcher or a running Slurm job, rather than proving
that integration is currently advancing. Local detection recognizes Python
processes running this checkout's `mains/local.py`, including `--stage`, through
Linux `/proc`. Slurm discovery uses one current-user query per snapshot and
matches both this project directory and its job-name prefix. A Slurm query
times out after five seconds; unavailable scheduler or process information is
shown explicitly. `UNFINISHED` means results remain incomplete without a known
active launcher/job; it does not establish failure. `UNREADABLE` calls for
inspection of the reported files. A checkpoint's presence does not establish
that it can be loaded: recovery still validates it on the execution platform.

**Progress and ETA refer to the shown stage, not the whole protocol.** Percentage
uses `Step / nstep`, and simulated nanoseconds use `Step × dt / 1000` from that
stage's input. CSV simulation time can include earlier stages and is not used
for progress. Recovery can roll the last reported step back to its checkpoint.
Old reports from before a new running execution are identified rather than
presented as measurements from the new execution.

The approximate stage ETA uses the latest finite positive speed in a bounded
CSV tail, without borrowing a speed from before OpenMM's startup marker.
During recovery, a freshly rewritten CSV can contain only retained old rows.
When a recent recovery backup identifies that prefix, speed and ETA wait until
a speed-bearing step exceeds the pre-recovery report. Progress still shows the
retained checkpoint steps. ETA is shown only for a confirmed running simulation
with usable measurements, and assumes the reported speed continues. It excludes later
stages, queue time, and setup/finalization work. Slurm's remaining allocation
time is a separate scheduler value. Report age is the CSV file's modification
age, not a process heartbeat or an exact timestamp for its last valid row.

Completion requires readable, nonempty `final_state.xml` and
`final_coordinates.pdb` files and no checkpoint in every scheduled stage.
Finishing one selected local stage does not complete the full protocol.
Status reads bounded CSV headers/tails and file metadata; it does not load
checkpoints, read trajectories, create files, modify locks, submit/cancel jobs,
or stop simulations. Ctrl+C affects only the status command.

Submission, status, and runtime output use restrained colours on supported
terminals. `NO_COLOR` or `TERM=dumb` disables colour; redirected output and
Slurm log files contain plain text. Submission shows grouped check results,
resources, the advisory start estimate, and the actual accepted job ID. Node
tables and repetitive setup diagnostics stay at DEBUG; actionable warnings and
failed-command diagnostics, including failed advisory checks, remain visible
automatically, without extra flags.

## SIMCALC GPU jobs

Consult the [SIMCALC cluster guide](http://salses.upc.es/SIMCON/simcalc.html)
for Slurm usage and cluster policies. Upload the project scripts and the
selected system's input bundle, and install the project's NVIDIA environment
in its shared `.venv` before using the submission command above.

Replace `SYSTEM_ID` with the exact input system/replica folder name.
**The same command starts a fresh run or recovers an interrupted run automatically.**
It performs the prerequisite checks and submits one job when work remains.
There is no separate recovery flag to remember:

```bash
python3 mains/simcalc.py "abeta42_7q4m_8bfa_replica_2"
```

Any failed prerequisite stops the launcher with an error before submission.

The system name selects matching input and output directories under
`data/inputs/openmm_native/solution/` and `data/outputs/openmm_native/solution/`.
Existing results select the action:

| Results | Action |
| --- | --- |
| Output directory absent | Submit a fresh run |
| Interrupted stage has a checkpoint and both reporter files | Submit recovery; validate the checkpoint on the allocated GPU |
| Earlier stages complete; next stage absent | Continue from the previous stage's final state |
| All stages complete | Report completion without submitting |
| Missing recovery files, inconsistent stage order, or empty/unrecognized output | Stop for inspection and preserve results |

A checkpoint takes precedence over final files, including an interruption during
finalization. The launcher refuses execution locks and duplicate queued/running
jobs for that system. It never removes stale locks or installs packages. Output
checks repeat before submission and under the execution lock on the GPU node.
The selected fresh/recovery mode travels with the job: a queued fresh job refuses
newly appearing results, and a recovery job refuses missing results (including
its selected checkpoint stage) instead of starting over. A recovery job whose
stages have since finished exits successfully.

Before submission, the launcher checks files, writable destinations, Slurm
connectivity, duplicate jobs, partition limits, and resource capacity. It can
create parent/log directories and temporary write probes during these checks.
Reported filesystem space does not establish your account's quota or guarantee
enough space for the whole run or recovery backups. Slurm's start-time prediction is advisory: a
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

Logs are saved as `mains/logs/SYSTEM_ID-JOB_ID.log`. The launcher prints the job
ID, log path, and `python3 mains/status.py` for checking progress. Owned locks are
released on normal completion or failure; interruption stops the active child
before releasing its lock. A forced kill or node crash can leave a lock for
manual inspection. If submission is interrupted or returns an unrecognized
reply, check `squeue` before retrying: Slurm might already have accepted it.

`mains/simcalc.py` delegates submission to `mains/_simcalc/submit.py`.
Inside that internal folder, `common.py` owns checks and process handling,
`run_solution.slurm` holds resource requests and loads CUDA, and
`run_solution.py` manages the allocated job. It runs `validate_solution.py`
in the project environment before invoking `mains/local.py solution SYSTEM_ID`
with that same interpreter. Both environments therefore use the same protocol
dispatch logic.

### Submit from PyCharm

These instructions follow PyCharm Pro 2026.2. Deployment uploads the code; a
Remote SSH External Tool runs the submission launcher on the SIMCALC login node.

1. Keep the existing **SIMCALC** SFTP Deployment configuration under
   **Settings → Build, Execution, Deployment → Deployment**. Its connection uses
   `carlos@simcalc.upc.es`, with root path `/nfs/carlos`. Under **Mappings**, the
   local project folder maps to deployment path `/on-boarding-phospholipids-openmm`.
   Together these select `/nfs/carlos/on-boarding-phospholipids-openmm` remotely.
2. Before uploading, keep `data`, `.venv`, `.git`, `.idea`, `mains/logs`, and
   any legacy `simcalc/logs`
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
| Arguments | `-u /nfs/carlos/on-boarding-phospholipids-openmm/mains/simcalc.py SYSTEM_ID` |
| Working directory | `/nfs/carlos/on-boarding-phospholipids-openmm` |
| SSH Configuration | The existing connection for `carlos@simcalc.upc.es` |

Replace `SYSTEM_ID` in Arguments with the replica's folder name. Under
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

### Migrating older launch commands

The four `mains/run_*` scripts and `simcalc/main.py` have been replaced; there
are no compatibility wrappers. Update saved commands and PyCharm configurations
to the two launchers above. Local commands now select `membrane` or `solution`
explicitly, and a single-stage run uses `--stage STAGE_NAME`.

Wait until all queued/running jobs using the old layout have finished before
replacing shared project files on SIMCALC. Deployment can leave removed Python
files or older `simcalc/main.sh` and `simcalc/common.sh` scripts on the server.
Remove obsolete scripts manually after that point; do not enable broad remote
deletion. Preserve existing `simcalc/logs/` files and all simulation outputs.
New submissions write logs and submission locks under `mains/logs/`; execution
locks remain alongside their system's output directory.

## Runtime Notes

- Every stage writes `checkpoint.chk` in its output directory at the same
  interval as trajectory frames. Computation since the last saved checkpoint
  is repeated on recovery.
- Rerun the same full-protocol command after a crash, locally or on SIMCALC,
  to skip completed stages, restore the interrupted stage, and run its remaining
  steps without repeating minimization or generating new velocities. A missing
  next stage starts from the previous stage's final dynamic state. Local
  `--stage` also restores a checkpoint; explicitly selecting a completed stage
  without one retains the existing behavior of rerunning that stage.
- After checkpoint loading succeeds, recovery validates both reporter files
  through the checkpoint step. It retains CSV records and DCD frames through
  that step, removes any later tail, corrects DCD frame-count and timing metadata, and then
  appends to the same `state_data.csv` and `trajectory.dcd`. Missing or corrupt
  data within the required prefix stops recovery before either file is changed.
  An incomplete trailing record is discarded only after the prefix is validated.
- Before replacing an affected reporter file, recovery finishes backups of all
  affected originals in a unique `recovery-*` directory inside that stage. Logs
  report the backup location, resumed step, and remaining steps. Repair uses
  streaming I/O and atomic file replacements; it keeps the checkpoint unchanged
  so retrying after interruption between replacements is safe. Allow disk space
  for the original-file backups and prepared replacements. Backups, including
  any partially prepared recovery directories, remain available for inspection.
- Incomplete stage directories without checkpoints are never overwritten by
  the shared workflow. Inspect such results manually. A full protocol also
  rejects results that violate stage order before starting any stage.
- A checkpoint is removed only after `final_state.xml` and
  `final_coordinates.pdb` are written successfully. If checkpoint loading
  fails, the run stops without overwriting partial outputs.
- Keep inputs, protocol settings, and the OpenMM environment unchanged during
  recovery. [OpenMM binary checkpoints](https://docs.openmm.org/latest/api-python/generated/openmm.app.checkpointreporter.CheckpointReporter.html)
  require compatible hardware and software as well as the same system. A
  checkpoint that fails to load on the allocated GPU stops recovery without
  replacing results. Completed stages use the portable `final_state.xml` to
  initialize the next stage. Portable-state fallback for interrupted stages,
  automatic Slurm resubmission, and extending completed runs are not supported.
- On Linux AMD hosts with ROCm installed, the package preloads the HIP runtime
  before OpenMM import so `HIP` can be used when available.
- On NVIDIA hosts, normal OpenMM `CUDA` selection remains unchanged.
- On CPU-only hosts, the platform fallback order still reaches `CPU`.
- Use `OPENMM_DEVICE_INDEX` to pin a specific GPU when multiple devices are
  visible.
- Use `CHARMM_GUI_MD_OPENMM_SKIP_BOOTSTRAP=1` to skip runtime bootstrap.
