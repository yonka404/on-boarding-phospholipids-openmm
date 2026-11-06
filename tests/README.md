# Test Organization

Tests are grouped by the production area they exercise, mirroring
`src/charmm_gui_md/` where possible:

- `runtime/`: package import and runtime bootstrap behavior.
- `inputs/`: OpenMM-native input adapter validation.
- `protocols/`: stage protocol parsing, schedule behavior, and related protocol
  models.
- `simulation/`: OpenMM simulation setup, platform selection, initialization, and
  reporter behavior.
- `workflows/`: orchestration code that coordinates inputs, protocols,
  simulations, and outputs.

Prefer the folder for the closest public behavior under test. If a test crosses
several production packages, place it under `workflows/` when it validates a user
workflow, otherwise place it beside the main component that owns the decision.

Keep using Python's built-in `unittest`. Nested test folders need `__init__.py`
files so `python -m unittest` can discover them.

Recovery coverage includes fake Slurm commands and temporary output directories
in `workflows/test_simcalc_launchers.py`, CSV/DCD corruption and interrupted
repair in `simulation/test_recovery.py`, and a tiny real OpenMM Reference-platform
checkpoint/resume in `workflows/test_checkpoint_recovery.py`. These tests do not
submit jobs, use production inputs, or run production simulations.

Status coverage in `workflows/test_status.py` uses temporary reports, fake
scheduler/process information, and a sleeping Python launcher fixture (no MD).
It exercises stage progress, recovery rollback, bounded reads, discovery,
table/stacked formatting, and a system-Python snapshot that must create no
files or caches. `runtime/test_console.py` covers colour selection and plain
runtime logs; launcher tests cover concise submission and captured failure
diagnostics. Local input bundles are not needed by these tests.
