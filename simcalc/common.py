"""Shared launcher checks. Keep this module compatible with system Python 3.8."""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import FrameType

logger = logging.getLogger(__name__)


class LauncherError(RuntimeError):
    """A failed prerequisite with an actionable message for the operator."""


class Interrupted(Exception):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"Interrupted by {signal.Signals(signum).name}.")


class SolutionPaths:
    def __init__(self, project_root: Path, system_id: str) -> None:
        self.project_root = project_root.resolve()
        self.system_id = system_id
        self.input_dir = (
            self.project_root
            / "data/inputs/openmm_native/solution"
            / system_id
            / "openmm"
        )
        self.output_root = self.project_root / "data/outputs/openmm_native/solution"
        self.output_dir = self.output_root / system_id
        self.execution_lock = self.output_root / f".{system_id}.lock"
        self.log_root = self.project_root / "simcalc/logs"
        self.submission_lock = self.log_root / f".{system_id}.submit.lock"
        self.job_name = f"solution-md-{system_id}"
        self.batch_file = self.project_root / "simcalc/run_solution.slurm"
        # Do not resolve this symlink: Python must recognize the virtual environment.
        self.python = self.project_root / ".venv/bin/python"


def system_name(value: str) -> str:
    if value in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError(
            "Provide one system ID containing only letters, digits, _, . or -."
        )
    return value


def check_files(paths: SolutionPaths) -> None:
    for name in (
        "pyproject.toml",
        "uv.lock",
        "mains/run_solution_sweep.py",
        "simcalc/common.py",
        "simcalc/run_solution.py",
        "simcalc/validate_solution.py",
        "simcalc/run_solution.slurm",
    ):
        path = paths.project_root / name
        if not path.is_file() or not os.access(path, os.R_OK):
            raise LauncherError(
                f"Missing or unreadable project file: {path}. Upload the project first."
            )
    if not paths.python.is_file() or not os.access(paths.python, os.X_OK):
        raise LauncherError(
            "Missing executable .venv/bin/python. Install the NVIDIA environment on a compatible GPU node."
        )
    for name in (
        "step3_input.psf",
        "step3_input.pdb",
        "step3_input.crd",
        "step4_equilibration.inp",
        "step5_production.inp",
        "sysinfo.dat",
        "toppar.str",
    ):
        path = paths.input_dir / name
        if (
            not path.is_file()
            or not os.access(path, os.R_OK)
            or path.stat().st_size == 0
        ):
            raise LauncherError(
                f"Missing, empty or unreadable input: {path}. Transfer the complete input bundle."
            )


def check_output(paths: SolutionPaths) -> None:
    if paths.output_dir.exists() or paths.output_dir.is_symlink():
        raise LauncherError(
            f"Output already exists: {paths.output_dir}. Inspect it before choosing a new replica ID "
            "or a recovery procedure."
        )


def check_fresh(paths: SolutionPaths) -> None:
    check_output(paths)
    if paths.execution_lock.exists() or paths.execution_lock.is_symlink():
        raise LauncherError(
            f"Execution lock exists: {paths.execution_lock}. Check squeue before considering manual cleanup."
        )


def check_writable(paths: SolutionPaths) -> None:
    # Write a small temporary file: permission bits alone cannot detect a full quota.
    for directory in (paths.output_root, paths.log_root):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=".write-check."
            ) as probe:
                probe.write(b"SIMCALC write check\n")
                probe.flush()
                os.fsync(probe.fileno())
        except OSError as exc:
            raise LauncherError(
                f"Cannot write to {directory}: {exc}. Check permissions, filesystem space and quota."
            ) from exc


@contextmanager
def directory_lock(path: Path, label: str) -> Iterator[None]:
    # mkdir is atomic across launchers, including those using the shared NFS disk.
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise LauncherError(
            f"{label} lock exists: {path}. Check for another job or launcher before manual cleanup."
        ) from exc
    try:
        yield
    finally:
        try:
            path.rmdir()
        except OSError as exc:
            logger.warning("Could not remove owned lock %s: %s", path, exc)


@contextmanager
def interruption_handlers() -> Iterator[None]:
    def interrupt(signum: int, frame: FrameType | None) -> None:
        raise Interrupted(signum)

    previous = {
        sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def defer_interruptions() -> Iterator[list[int]]:
    pending: list[int] = []

    def record(signum: int, frame: FrameType | None) -> None:
        pending.append(signum)

    previous = {
        sig: signal.signal(sig, record) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield pending
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_command(
    arguments: Sequence[str],
    failure: str,
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    # Argument lists avoid shell quoting. Child diagnostics go straight to the console/log.
    with ExitStack() as cleanup:
        child = None
        try:
            # Record signals during spawn until we own the child and can stop it.
            with defer_interruptions() as pending:
                child = cleanup.enter_context(
                    subprocess.Popen(
                        arguments,
                        stdout=subprocess.PIPE if capture else None,
                        text=True,
                    )
                )
            if pending:
                raise Interrupted(pending[0])
            stdout, _ = child.communicate()
        except Interrupted as exc:
            if child is None:
                raise
            # Keep the execution lock until the child is stopped, even on repeated signals.
            previous = {
                sig: signal.signal(sig, signal.SIG_IGN)
                for sig in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                child.send_signal(exc.signum)
                try:
                    child.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.communicate()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            raise
    result = subprocess.CompletedProcess(arguments, child.returncode, stdout or "")
    if check and result.returncode != 0:
        raise LauncherError(
            f"{failure} (exit {result.returncode}). Review the diagnostic above."
        )
    return result


def memory_mb(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)([MG]?)", value)
    if not match:
        raise LauncherError(
            f"Unsupported memory setting: {value}. Use whole MiB or a number followed by M or G."
        )
    return int(match[1]) * (1024 if match[2] == "G" else 1)


def time_seconds(value: str) -> int:
    if value in {"UNLIMITED", "INFINITE"}:
        return 2147483647
    match = re.fullmatch(r"(?:(\d+)-)?(\d+(?::\d+){0,2})", value)
    if not match:
        raise LauncherError(f"Unsupported Slurm time: {value}.")
    parts = [int(part) for part in match[2].split(":")]
    # With a day prefix, the remaining fields start at hours; otherwise at minutes.
    if match[1] is not None:
        parts += [0] * (3 - len(parts))
    else:
        parts = [0] * (3 - len(parts)) + parts
        if ":" not in match[2]:
            parts = [0, int(match[2]), 0]
    hours, minutes, seconds = parts
    return int(match[1] or 0) * 86400 + hours * 3600 + minutes * 60 + seconds
