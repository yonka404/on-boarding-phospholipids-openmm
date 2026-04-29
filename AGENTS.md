# AGENTS.md

## Repository Overview

This is a Python/OpenMM scaffold for protein-in-membrane molecular dynamics runs.

Important paths:
- `src/protein_membrane_md/`: package source
- `src/protein_membrane_md/inputs/`: input format adapters
- `src/protein_membrane_md/simulation/`: OpenMM setup, platform selection, initialization, and reporters
- `src/protein_membrane_md/workflows/`: stage and sweep runners
- `mains/`: runnable scripts
- `tests/`: test suite written with Python's built-in `unittest`
- `data/`: local input data; do not modify unless explicitly requested

## Input Data Context

The files under `data/inputs/` come from the CHARMM-GUI web platform, a system builder for preparing molecular dynamics simulations. This repo separates CHARMM-GUI-provided formats into explicit input adapters so each supported format has clear validation and runtime behavior.

The durable input format folders are `data/inputs/charmmgui/` and `data/inputs/openmm_native/`. These are the CHARMM-GUI exports expected to be most compatible with this Python/OpenMM project. Prefer the native OpenMM format as the default runtime input format when both formats are available.

CHARMM-GUI can also export files for NAMD, GROMACS, TINKER, and other engines. Those formats are not expected to be used in this repo because they are less directly compatible with an OpenMM-focused Python workflow. Do not add support for them unless the user explicitly asks for it.

The `data/outputs/` directory should mirror the `data/inputs/` format structure. Results produced from `data/inputs/openmm_native/` should be written under `data/outputs/openmm_native/`; results produced from `data/inputs/charmmgui/` should be written under `data/outputs/charmmgui/`. Preserve that mapping so simulation results can be traced back to the input format that generated them.

Input directory validation is implemented with Pydantic in `src/protein_membrane_md/inputs/`. Preserve or extend those validation models when changing expected input files or formats.

## Environment

Use Python 3.12.10 or compatible. Prefer `uv` for environment and test commands.

Declared project dependencies are OpenMM, Pydantic, Ruff, and mypy. Optional OpenMM platform packages are declared for AMD/HIP and NVIDIA/CUDA. Do not add or recommend dependencies that are not actually intended for this project.

Prefer Python standard library tools when they fit the task. Use Python's built-in `logging` module for logging and `unittest` for tests. Do not introduce third-party logging or test libraries such as `loguru` or `pytest` unless the user explicitly asks for them.

## Checks

Before finishing code changes, run the most relevant tests:

```bash
uv run python -m unittest
```

For narrower changes, run the matching test module first, for example:

```bash
uv run python -m unittest tests.test_stage_protocol
```

Use these checks when relevant to the changed code:

```bash
uv run ruff check .
uv run mypy src
```

## Engineering Rules

- Keep changes small and behavior-preserving unless the task explicitly asks for a behavior change.
- Preserve OpenMM platform fallback behavior unless the task is specifically about platform selection.
- Do not run long simulations or sweeps unless explicitly requested.
- Do not commit generated trajectories, checkpoints, logs, or large local data outputs.
- Do not add dependencies without explaining why they are needed and confirming they belong in this project.
- Add logs only when they help operators or developers understand runtime behavior, diagnose failures, or audit important decisions; avoid noisy logs that repeat obvious control flow.
- Choose the lowest standard logging level that accurately reflects the event: `debug` for detailed diagnostics, `info` for useful normal progress or configuration choices, `warning` for recoverable unexpected conditions, `error` for failed operations that prevent the requested action from completing, and `critical` only for unrecoverable process-level failures.
- Do not push changes to a remote branch while working unless the user explicitly asks for a push or pull request.
- Preserve or add focused tests for changes to input parsing, stage protocol behavior, platform fallback, or workflow orchestration.

## Done Means

A change is complete only when:
- Relevant tests or checks have been run, or any skipped check is clearly explained.
- Changed runtime behavior is covered by a test or a documented reason.
- The final response states what changed and what verification was run.
