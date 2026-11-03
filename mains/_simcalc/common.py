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
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from mains._common import system_paths

logger = logging.getLogger(__name__)
SOLUTION_STAGES = ("step4_equilibration", "step5_production")


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
        inputs, self.output_dir = system_paths(
            "solution", system_id, project_root=self.project_root
        )
        self.input_dir = inputs / "openmm"
        self.output_root = self.output_dir.parent
        self.execution_lock = self.output_root / f".{system_id}.lock"
        self.log_root = self.project_root / "mains/logs"
        self.submission_lock = self.log_root / f".{system_id}.submit.lock"
        self.job_name = f"solution-md-{system_id}"
        self.batch_file = self.project_root / "mains/_simcalc/run_solution.slurm"
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
        "mains/__init__.py",
        "mains/_common.py",
        "mains/local.py",
        "mains/_simcalc/__init__.py",
        "mains/_simcalc/common.py",
        "mains/_simcalc/run_solution.py",
        "mains/_simcalc/validate_solution.py",
        "mains/_simcalc/run_solution.slurm",
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


@dataclass(frozen=True)
class OutputAction:
    mode: str
    stage: str | None = None
    checkpoint_stage: str | None = None


def check_output(
    paths: SolutionPaths,
    expected_mode: str | None = None,
    expected_checkpoint_stage: str | None = None,
) -> OutputAction:
    """Classify files only; OpenMM validates checkpoints on the allocated GPU."""
    output = paths.output_dir
    exists = output.exists() or output.is_symlink()
    if expected_mode == "fresh" and exists:
        raise LauncherError(
            f"Output already exists: {output}. This job was submitted as fresh; inspect the results."
        )
    if not exists:
        if expected_mode == "resume":
            raise LauncherError(
                f"Recovery results are missing: {output}. Inspect the paths; refusing to start over."
            )
        return OutputAction("fresh", "step4_equilibration")

    def reject(reason: str) -> None:
        raise LauncherError(f"Cannot recover {output}: {reason}. Inspect results before retrying.")

    if output.is_symlink() or not output.is_dir():
        reject("output is not a regular directory")
    stages = SOLUTION_STAGES
    children = {path.name for path in output.iterdir()}
    if expected_checkpoint_stage is not None and expected_checkpoint_stage not in children:
        reject(f"selected checkpoint stage {expected_checkpoint_stage} is missing; refusing to start over")
    if not children or children - set(stages):
        reject("empty or unrecognized output directory")

    def usable(path: Path) -> bool:
        return (
            not path.is_symlink() and path.is_file()
            and os.access(path, os.R_OK) and path.stat().st_size > 0
        )

    next_stage = None
    checkpoint_stage = None
    for name in stages:
        stage = output / name
        if name not in children:
            if next_stage is None:
                next_stage = name
            continue
        if next_stage is not None:
            reject(f"inconsistent stage order: {name} exists after incomplete {next_stage}")
        if stage.is_symlink() or not stage.is_dir():
            reject(f"{name} is not a regular stage directory")

        checkpoint = stage / "checkpoint.chk"
        if checkpoint.exists() or checkpoint.is_symlink():
            for filename in ("checkpoint.chk", "state_data.csv", "trajectory.dcd"):
                if not usable(stage / filename):
                    reject(f"{name} needs a nonempty, readable {filename} for checkpoint recovery")
            next_stage = name
            checkpoint_stage = name
        elif not all(usable(stage / filename) for filename in ("final_state.xml", "final_coordinates.pdb")):
            reject(f"incomplete {name} has no checkpoint; refusing to overwrite it")
    return OutputAction("complete") if next_stage is None else OutputAction("resume", next_stage, checkpoint_stage)


def check_available(
    paths: SolutionPaths,
    expected_mode: str | None = None,
    expected_checkpoint_stage: str | None = None,
) -> OutputAction:
    if paths.execution_lock.exists() or paths.execution_lock.is_symlink():
        raise LauncherError(
            f"Execution lock exists: {paths.execution_lock}. Check squeue before considering manual cleanup."
        )
    return check_output(paths, expected_mode, expected_checkpoint_stage)


def log_action(paths: SolutionPaths, action: OutputAction) -> None:
    if action.mode == "complete":
        logger.info("System %s is complete; no simulation needed", paths.system_id)
    else:
        logger.info("System %s: %s from %s", paths.system_id, action.mode, action.stage)


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
