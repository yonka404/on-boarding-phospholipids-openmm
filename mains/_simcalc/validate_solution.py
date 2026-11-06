"""Check the Slurm GPU allocation and report the simulation startup settings."""

import argparse
import os
import platform
import re
import socket
import sys
from importlib import metadata
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mains._console import Console


def validate_solution(system_id: str) -> None:
    """Check allocation metadata; the simulation owns input and CUDA validation."""
    if sys.version_info < (3, 12, 10):
        raise RuntimeError("Python 3.12.10 or newer is required; rebuild the project .venv.")

    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdecimal():
        raise ValueError("SLURM_JOB_ID is missing or invalid; submit through Slurm.")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9_./-]+)", visible_gpu):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must identify exactly one Slurm-assigned GPU; "
            "do not override Slurm's GPU visibility."
        )

    from mains._common import system_id_argument, system_paths

    system_id = system_id_argument(system_id)
    _, outputs_dir = system_paths("solution", system_id)
    console = Console()
    console.field("Allocation", f"Validated system={system_id} job={job_id} host={socket.gethostname()}", tone="green")
    console.field("Environment", f"Python={platform.python_version()} OpenMM={metadata.version('openmm')}")
    console.field("Platform", f"requested_platform=CUDA device=0 (Slurm GPU {visible_gpu})")
    console.field("Output", outputs_dir.resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system_id")
    args = parser.parse_args(argv)
    try:
        validate_solution(args.system_id)
    except (ImportError, OSError, RuntimeError, ValueError, argparse.ArgumentTypeError) as exc:
        Console(sys.stderr).field("ERROR", f"GPU allocation: {exc}", tone="red")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
