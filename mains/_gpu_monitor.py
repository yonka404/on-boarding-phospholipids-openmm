"""Report this MD process's GPU usage to the login node through the job log directory."""

from __future__ import annotations

import json
import logging
import math
import os
import stat
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from mains._gpu import GpuUsage, measure_gpu

logger = logging.getLogger(__name__)
SAMPLE_INTERVAL = 30
MAX_SAMPLE_AGE = 90


def _sample_path(logs: Path, system_id: str, job_id: str) -> Path:
    return logs / f"{system_id}-{job_id}.gpu.json"


def _publish(path: Path, run_started_at: float, usage: GpuUsage) -> None:
    sample = dict(asdict(usage), run_started_at=run_started_at, sampled_at=time.time())
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".gpu-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(sample, stream, allow_nan=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _poll(stop: threading.Event, path: Path | None) -> None:
    pid = os.getpid()
    run_started_at = time.time()
    warned_multiple = warned_error = published = False
    while not stop.is_set():
        try:
            usage = measure_gpu(pid)
            if usage.count is not None and usage.count > 1 and not warned_multiple:
                logger.warning("Simulation is using %d GPUs. One GPU per simulation is expected.", usage.count)
                warned_multiple = True
            if path is not None and not stop.is_set():
                _publish(path, run_started_at, usage)
                published = True
        except (OSError, ValueError) as exc:
            # Monitoring is optional. A failed driver query or NFS write must not interrupt MD.
            if not warned_error:
                logger.warning("GPU monitoring unavailable (%s). The simulation continues.", exc)
                warned_error = True
        stop.wait(SAMPLE_INTERVAL)
    if published and path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Could not remove GPU sample %s: %s", path, exc)


@contextmanager
def monitor_gpu(logs: Path, kind: str, system_id: str) -> Iterator[None]:
    """Warn once locally, and publish samples only inside the matching SIMCALC job."""
    job_id = os.environ.get("SLURM_JOB_ID", "")
    path = None
    if kind == "solution" and job_id.isdecimal() and os.environ.get("SLURM_JOB_NAME") == f"solution-md-{system_id}":
        path = _sample_path(logs, system_id, job_id)
    stop = threading.Event()
    thread = threading.Thread(target=_poll, args=(stop, path), name="gpu-monitor", daemon=True)
    try:
        thread.start()
    except RuntimeError as exc:
        logger.warning("Could not start GPU monitoring (%s). The simulation continues.", exc)
        yield
        return
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)  # A slow GPU utility or filesystem must not hold up the job.


def read_job_usage(
    logs: Path, system_id: str, job_id: str, started_at: float | None, now: float | None,
) -> GpuUsage:
    """Read only the current allocation's sample, including after Slurm requeues a job ID."""
    if started_at is None or not job_id.isdecimal():
        return GpuUsage(problem="Cannot match a GPU sample to this allocation.")
    path = _sample_path(logs, system_id, job_id)
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError
        with path.open("rb") as stream:
            raw = stream.read(4097)
        if len(raw) > 4096:
            raise ValueError
        sample = json.loads(raw)
        run_start, sampled = sample["run_started_at"], sample["sampled_at"]
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (run_start, sampled)) or run_start > sampled:
            raise ValueError
        if run_start < started_at:
            return GpuUsage(problem="Waiting for a GPU sample from the restarted job.")
        age = (time.time() if now is None else now) - sampled
        if age < 0 or age > MAX_SAMPLE_AGE:
            return GpuUsage(problem=f"GPU sample is outside the last {MAX_SAMPLE_AGE} seconds.")
        count, memory, problem = sample["count"], sample["memory_bytes"], sample["problem"]
        if any(v is not None and (type(v) is not int or not 0 <= v < 2**63) for v in (count, memory)):
            raise ValueError
        if count == 0 and memory:
            raise ValueError
        if problem is not None and not isinstance(problem, str):
            raise ValueError
        if (count is None or memory is None) and not problem:
            raise ValueError
        return GpuUsage(count, memory, problem)
    except FileNotFoundError:
        return GpuUsage(problem="No GPU sample from this job yet.")
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return GpuUsage(problem="Cannot read this job's GPU sample.")
