# Test Organization

Tests are grouped by the production area they exercise, mirroring
`src/protein_membrane_md/` where possible:

- `runtime/`: package import and runtime bootstrap behavior.
- `inputs/`: input adapter validation for CHARMM-GUI and OpenMM-native files. Create
  this folder when adding the first input-adapter tests.
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
