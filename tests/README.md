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

## Running without production data

The default suite needs no `data/` directory and must not create one. Run it
from the repository root with the pinned Python 3.12.10 environment:

```bash
uv sync --locked --python 3.12.10
uv run --locked python -m unittest -v
uv run --locked ruff check .
uv run --locked mypy src
```

Small synthetic OpenMM-native protocol examples live in
`fixtures/openmm_native/{membrane,solution}/`. Tests resolve these paths from
`__file__` and enumerate the expected files explicitly: missing or empty
fixtures fail. These are parser examples, not molecular input bundles.
Input-validation trees, malformed files, reports, trajectories, and checkpoints
are created in `TemporaryDirectory` and cleaned automatically. Tests checking
input validation mock molecular-file readers; integration checks construct tiny
OpenMM systems on the Reference platform.

Tests must scope environment changes with `mock.patch.dict` and mutable
bootstrap state with `mock.patch.object`, restoring both after each test.
Do not set bootstrap flags during test-module import. Platform fallback tests
discard inherited `OPENMM_*` overrides within their own scope; the opt-in GPU
test keeps its requested backend. Fake-launcher environments remove exported
Bash functions, startup hooks, and inherited Bash options so a login shell's
`module` function cannot replace the test executable. Real Slurm allocations
retain their `CUDA_VISIBLE_DEVICES` setting.

Verify discovery from another working directory with explicit paths:

```bash
test_repo=$PWD
(cd /tmp && "$test_repo/.venv/bin/python" -m unittest discover \
    -s "$test_repo/tests" -t "$test_repo" -v)
```

For acceptance, use a clean checkout of the test changes with no production
data or reused environment. Check that `data/` is absent before and after the
suite; leave any production checkout and environment intact. Repeat the suite
with inherited settings, for example:

```bash
test ! -e data
env OPENMM_PLATFORM=CUDA OPENMM_DEVICE_INDEX=7 OPENMM_OPENCL_PLATFORM_INDEX=3 \
    'BASH_FUNC_module%%=() { return 91; }' .venv/bin/python -m unittest -v
test ! -e data
```

Check reversed module order in a fresh process as well:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import unittest

modules = sorted((".".join(p.with_suffix("").parts)
                  for p in Path("tests").rglob("test*.py")), reverse=True)
suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

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

## SIMCALC verification

Prepare a separate test checkout/upload, including `tests/fixtures`, the source,
launchers, `pyproject.toml`, `uv.lock`, and `.python-version`. Do not upload
`data/` or a local `.venv`. Keep queued/running jobs on their existing checkout
and environment. Use PyCharm's SIMCALC SSH session or an authenticated SSH
connection to the login node, then request a short allocation:

```bash
squeue -u "$USER"
srun -p gpu --gres=gpu:1 --cpus-per-task=1 --mem=2G --time=00:15:00 --pty bash -l
```

Inside that allocation, change to the separate test checkout. Load CUDA and
build its own environment there, never in the production checkout:

```bash
module load cuda-12.9
hostname
getconf GNU_LIBC_VERSION  # must be glibc 2.28+
test ! -e data
uv sync --locked --python 3.12.10 --extra nvidia
.venv/bin/python - <<'PY'
import sys
from importlib.metadata import version
import openmm
import openmm.app
import pydantic

assert sys.version_info[:3] == (3, 12, 10), sys.version
print(sys.executable, sys.version)
for package in ("openmm", "openmm-cuda-12", "pydantic", "numpy"):
    installed = version(package)
    print(package, installed)
    if package in {"openmm", "openmm-cuda-12"}:
        assert installed == "8.3.1", (package, installed)
print("OpenMM imports OK")
PY
.venv/bin/python -m unittest -v
test ! -e data
```

Record the node name, date, interpreter/package versions, suite result, and
continued absence of `data/`. Repeat discovery and contamination checks above
in this environment. A local pass does not establish compute-node binary or
CUDA compatibility. Unavailable allocations or authentication leave the
compute-node check pending.

The GPU smoke test is skipped by default. Run it explicitly with
`OPENMM_GPU_SMOKE=1 OPENMM_PLATFORM=OpenCL uv run --locked python -m unittest tests.simulation.test_gpu_smoke -v`
on AMD, or use `OPENMM_PLATFORM=CUDA` in a SIMCALC GPU allocation with the
NVIDIA extra installed. It runs five periodic PME steps, checks finite energy,
and requires the requested GPU backend. Record each node's CUDA smoke result
separately from its default-suite result; unavailable node checks stay pending.
See README.md for per-node commands.
