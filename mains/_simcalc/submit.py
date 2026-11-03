"""Check all submission prerequisites, then submit exactly one SIMCALC solution job."""

from __future__ import annotations

import argparse
import logging
import os
import pwd
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mains._common import PROJECT_ROOT
from mains._simcalc.common import (
    Interrupted,
    LauncherError,
    SolutionPaths,
    check_available,
    check_files,
    check_writable,
    directory_lock,
    interruption_handlers,
    log_action,
    memory_mb,
    run_command,
    system_name,
    time_seconds,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Resources:
    partition: str
    cpus: int
    memory: int
    walltime: int

    @classmethod
    def from_batch_file(cls, path: Path) -> Resources:
        settings = dict(
            re.findall(r"^#SBATCH\s+--([\w-]+)=(\S+)", path.read_text(), re.MULTILINE)
        )
        try:
            resources = cls(
                settings["partition"],
                int(settings["cpus-per-task"]),
                memory_mb(settings["mem"]),
                time_seconds(settings["time"]),
            )
        except (KeyError, ValueError) as exc:
            raise LauncherError(
                f"Missing or invalid resource request in {path}."
            ) from exc
        if (
            not resources.partition
            or min(resources.cpus, resources.memory, resources.walltime) <= 0
        ):
            raise LauncherError(f"Invalid partition or resource request in {path}.")
        if any(
            settings.get(key) != value
            for key, value in {"nodes": "1", "ntasks": "1", "gres": "gpu:1"}.items()
        ):
            raise LauncherError(
                f"{path} must request one node, one task and --gres=gpu:1."
            )
        return resources


def check_slurm(paths: SolutionPaths, resources: Resources) -> None:
    controller = run_command(
        ["scontrol", "ping"],
        "Slurm controller is unreachable; reconnect or contact SIMCALC support",
        capture=True,
    )
    if "is UP" not in controller.stdout:
        raise LauncherError(f"Slurm controller is not UP: {controller.stdout.strip()}")
    queue = run_command(
        ["squeue", "-h", "-u", pwd.getpwuid(os.getuid()).pw_name, "-o", "%i|%j|%T"],
        "Cannot query your Slurm jobs",
        capture=True,
    )
    for row in queue.stdout.splitlines():
        if not row.strip():
            continue
        try:
            job_id, job_name, state = row.strip().split("|", 2)
        except ValueError as exc:
            raise LauncherError(
                f"Cannot parse Slurm queue information: {row!r}. Check squeue output."
            ) from exc
        if job_name == paths.job_name:
            raise LauncherError(
                f"Job {job_id} is already {state} for {paths.system_id}. Do not submit it twice."
            )

    partition = run_command(
        ["scontrol", "show", "partition", resources.partition, "-o"],
        f"Cannot read partition {resources.partition}",
        capture=True,
    )
    fields = dict(
        field.split("=", 1) for field in partition.stdout.split() if "=" in field
    )
    if fields.get("State") != "UP":
        raise LauncherError(f"Partition {resources.partition} is not UP.")
    if "MaxTime" not in fields:
        raise LauncherError(
            f"Cannot determine partition {resources.partition} walltime limit."
        )
    if resources.walltime > time_seconds(fields["MaxTime"]):
        raise LauncherError(
            f"Requested walltime exceeds partition {resources.partition} limit ({fields['MaxTime']})."
        )

    nodes = run_command(
        ["sinfo", "-N", "-h", "-p", resources.partition, "-o", "%N|%T|%G|%c|%m"],
        "Cannot read GPU node availability",
        capture=True,
    )
    print(
        "GPU node availability (node | state | GPUs | CPUs | RAM MiB):\n"
        + nodes.stdout,
        flush=True,
    )
    if not any(
        node_has_capacity(row, resources)
        for row in nodes.stdout.splitlines()
        if row.strip()
    ):
        raise LauncherError(
            f"No usable node in {resources.partition} has capacity for one GPU, {resources.cpus} CPUs "
            f"and {resources.memory} MiB RAM. Check node health and resource requests."
        )


def node_has_capacity(row: str, resources: Resources) -> bool:
    try:
        _, state, gres, cpus, memory = row.strip().split("|")
        if re.search(r"down|drain|fail|invalid|unknown|maint", state, re.IGNORECASE):
            return False
        gpu_count = 0
        for resource in gres.split(","):
            parts = resource.split("(", 1)[0].split(":")
            if parts[0] == "gpu" and len(parts) in {2, 3}:
                gpu_count += int(parts[-1])
        return (
            gpu_count >= 1
            and int(cpus) >= resources.cpus
            and int(memory) >= resources.memory
        )
    except ValueError as exc:
        raise LauncherError(
            f"Cannot parse Slurm node information: {row!r}. Check sinfo output."
        ) from exc


def submit(system_id: str) -> None:
    paths = SolutionPaths(PROJECT_ROOT, system_id)
    os.chdir(paths.project_root)

    # Always check files and destinations before contacting Slurm; no OpenMM import here.
    check_files(paths)
    action = check_available(paths)
    log_action(paths, action)
    if action.mode == "complete":
        return
    check_writable(paths)

    for command in ("sbatch", "scontrol", "sinfo", "squeue"):
        if shutil.which(command) is None:
            raise LauncherError(
                f"Required command is missing: {command}. Run this on the SIMCALC login node."
            )
    resources = Resources.from_batch_file(paths.batch_file)

    # One launcher at a time for this replica, including while predicting and submitting.
    with directory_lock(paths.submission_lock, "Submission"):
        action = check_available(paths, action.mode, action.checkpoint_stage)
        if action.mode == "complete":
            log_action(paths, action)
            return
        check_slurm(paths, resources)
        print(
            "Filesystem space (this does not check your quota or predict trajectory size):",
            flush=True,
        )
        for directory in (paths.output_root, paths.log_root):
            usage = shutil.disk_usage(directory)
            print(
                f"{directory}: {usage.free / 1024**3:.1f} GiB available / {usage.total / 1024**3:.1f} GiB total",
                flush=True,
            )

        options = [
            f"--chdir={paths.project_root}",
            f"--job-name={paths.job_name}",
            f"--output={paths.log_root}/{system_id}-%j.log",
            str(paths.batch_file),
            *(["--checkpoint-stage", action.checkpoint_stage] if action.checkpoint_stage else []),
            "--mode",
            action.mode,
            system_id,
        ]
        # The estimate is advisory. Occupied nodes and missing predictions can still queue.
        estimate = run_command(
            ["sbatch", "--test-only", *options],
            "Start estimate unavailable",
            check=False,
        )
        if estimate.returncode != 0:
            logger.warning(
                "Start estimate unavailable. Actual submission will determine acceptance."
            )
        checked = check_available(paths, action.mode, action.checkpoint_stage)
        if checked.mode == "complete":
            log_action(paths, checked)
            return
        try:
            reply = run_command(
                ["sbatch", "--parsable", *options],
                "Slurm rejected submission",
                capture=True,
            )
        except (Interrupted, OSError):
            logger.warning(
                "Submission may have succeeded; inspect squeue before retrying."
            )
            raise
        job_id = reply.stdout.strip().split(";", 1)[0]
        if not re.fullmatch(r"[0-9]+", job_id):
            raise LauncherError(
                f"Slurm replied: {reply.stdout.strip()}. Submission may have succeeded; inspect squeue before retrying."
            )
        print(f"Submitted system {system_id} as job {job_id}.", flush=True)
        print(f"Log: {paths.log_root}/{system_id}-{job_id}.log")
        print(f"Status: squeue -j {job_id}")
        print(
            f"Follow log after the job starts: tail -f '{paths.log_root}/{system_id}-{job_id}.log'",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system_id", type=system_name)
    args = parser.parse_args(argv)
    try:
        with interruption_handlers():
            submit(args.system_id)
    except Interrupted as exc:
        logger.error("%s", exc)
        return 128 + exc.signum
    except (LauncherError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    return 0

