"""Recheck the allocated node, then run the selected solution in the project environment."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simcalc.common import (
    Interrupted,
    LauncherError,
    SolutionPaths,
    check_files,
    check_fresh,
    check_output,
    check_writable,
    directory_lock,
    interruption_handlers,
    run_command,
    system_name,
)

logger = logging.getLogger(__name__)


def run_job(system_id: str) -> int:
    submit_dir = os.environ.get("SLURM_SUBMIT_DIR", "")
    if not submit_dir or not re.fullmatch(
        r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")
    ):
        raise LauncherError(
            "Submit through simcalc/main.py; this script needs a Slurm allocation."
        )
    paths = SolutionPaths(Path(submit_dir), system_id)
    os.chdir(paths.project_root)

    # Files can change while queued. Recheck, then lock before touching replica output.
    check_files(paths)
    check_fresh(paths)
    check_writable(paths)
    with directory_lock(paths.execution_lock, "Execution"):
        check_output(paths)

        # Use system Python until glibc is checked; the OpenMM wheels need glibc 2.34+.
        libc = run_command(
            ["getconf", "GNU_LIBC_VERSION"],
            "Cannot determine node glibc version",
            capture=True,
        ).stdout.strip()
        match = re.fullmatch(r"glibc ([0-9]+)\.([0-9]+)", libc)
        if not match or (int(match[1]), int(match[2])) < (2, 34):
            raise LauncherError(
                f"Node has {libc!r}; OpenMM requires glibc 2.34+. Ask SIMCALC support about this node."
            )

        # Leave CUDA_VISIBLE_DEVICES intact. Device 0 is Slurm's first visible GPU.
        os.environ["OPENMM_PLATFORM"] = "CUDA"
        os.environ["OPENMM_DEVICE_INDEX"] = "0"
        run_command(
            [str(paths.python), "-u", "simcalc/validate_solution.py", system_id],
            "Allocation validation failed; the simulation was not started",
        )
        # The existing workflow validates molecular inputs and initializes real CUDA contexts.
        result = run_command(
            [str(paths.python), "-u", "mains/run_solution_sweep.py", system_id],
            "Solution sweep failed",
            check=False,
        )
        return result.returncode if result.returncode >= 0 else 128 - result.returncode


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system_id", type=system_name)
    args = parser.parse_args(argv)
    try:
        with interruption_handlers():
            return run_job(args.system_id)
    except Interrupted as exc:
        logger.error("%s", exc)
        return 128 + exc.signum
    except (LauncherError, OSError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
