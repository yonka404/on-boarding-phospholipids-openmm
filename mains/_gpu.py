"""Per-process NVIDIA and ROCm compute GPU accounting for launcher telemetry."""

from __future__ import annotations

import errno
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_KFD_PROC = Path("/sys/class/kfd/kfd/proc")


@dataclass(frozen=True)
class GpuUsage:
    count: int | None = None
    memory_bytes: int | None = None
    problem: str | None = None


def _unsigned(value: str) -> int | None:
    value = value.strip()
    if len(value) <= 20 and value.isascii() and value.isdecimal():
        return int(value)
    return None


def _total(values: Iterable[int | None]) -> int | None:
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total


def _nvidia_usage(pid: int) -> GpuUsage | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, encoding="utf-8", errors="replace", timeout=2, check=False,
    )
    if result.returncode == 9:  # nvidia-smi reports that the driver is unloaded.
        raise OSError(errno.ENODEV, "NVIDIA driver is not loaded")
    if result.returncode:
        raise ValueError(f"NVIDIA process query failed (exit {result.returncode})")

    devices: dict[str, int | None] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        row_pid = _unsigned(fields[1]) if len(fields) > 1 else None
        if not row_pid:
            raise ValueError("NVIDIA process list contains an unidentified PID")
        if row_pid != pid:
            continue
        device = fields[0]
        if not re.fullmatch(r"(GPU|MIG)-[a-zA-Z0-9/-]+", device):
            return GpuUsage(problem="NVIDIA GPU identifier unavailable")
        mib = _unsigned(fields[2]) if len(fields) == 3 else None
        memory = None if mib is None else mib * 1024**2
        if device in devices and devices[device] != memory:
            memory = None
        devices[device] = memory

    if not devices:
        return None
    memory = _total(devices.values())
    problem = "NVIDIA process memory unavailable or ambiguous" if memory is None else None
    return GpuUsage(len(devices), memory, problem)


def _read_counter(path: Path) -> int | None:
    try:
        return _unsigned(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError):
        return None


def _compute_queue_devices(directory: Path) -> set[int] | None:
    devices = set()
    try:
        for queue in directory.iterdir():
            kind = _read_counter(queue / "type")
            if kind not in (0, 1, 2, 3, 4):
                return None
            if kind == 0:  # KFD_QUEUE_TYPE_COMPUTE
                device = _read_counter(queue / "gpuid")
                if not device:
                    return None
                devices.add(device)
    except OSError:
        return None
    return devices


def _amd_usage(pid: int) -> GpuUsage | None:
    directory = _KFD_PROC / str(pid)
    try:
        entries = list(directory.iterdir())
    except FileNotFoundError:
        # A working KFD process list can also omit this PID.
        _KFD_PROC.stat()
        return None

    # KFD creates per-PID attributes for every device the process can see.
    # Only its compute queues and positive VRAM allocations demonstrate use.
    devices = set()
    vram: dict[int, int] = {}
    for entry in entries:
        prefix, _, suffix = entry.name.partition("_")
        if prefix not in ("vram", "sdma", "stats", "counters"):
            continue
        device = _unsigned(suffix)
        if not device:
            return GpuUsage(problem="invalid AMD KFD GPU identifier")
        devices.add(device)
        if prefix == "vram":
            memory = _read_counter(entry)
            if memory is not None:
                vram[device] = memory

    queues = _compute_queue_devices(directory / "queues")
    active = {device for device, memory in vram.items() if memory > 0}
    if queues is not None:
        devices.update(queues)
        active.update(queues)
    if not devices:
        return GpuUsage(problem="AMD KFD per-device accounting unavailable")
    missing_vram = devices - vram.keys()
    count: int | None = len(active)
    memory_bytes = sum(vram.values()) if not missing_vram else None
    problems = []
    if missing_vram:
        problems.append("AMD KFD VRAM unavailable")
    if missing_vram - active or (queues is None and devices != active):
        count = None
        problems.append("AMD KFD device count uncertain")
    return GpuUsage(count, memory_bytes, ", ".join(problems) or None)


def measure_gpu(pid: int) -> GpuUsage:
    """Measure devices and GPU memory the drivers attribute to this PID."""
    readings = []
    problems = []
    driver_queried = False
    for driver, query in (("NVIDIA", _nvidia_usage), ("AMD KFD", _amd_usage)):
        try:
            reading = query(pid)
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            problems.append(f"{driver} process query timed out")
        except OSError as exc:
            if exc.errno == errno.ENODEV:
                continue
            problems.append(f"{driver} process query failed ({exc.strerror or type(exc).__name__})")
        except ValueError as exc:
            problems.append(str(exc))
        else:
            driver_queried = True
            if reading is not None:
                readings.append(reading)
    if problems:
        # An inaccessible driver can hide another device used by this PID.
        return GpuUsage(problem=", ".join(problems))
    if not readings:
        reason = (
            f"PID {pid} not listed by GPU driver"
            if driver_queried else "nvidia-smi and AMD KFD process accounting unavailable"
        )
        return GpuUsage(problem=reason)
    problems.extend(reading.problem for reading in readings if reading.problem)
    return GpuUsage(
        _total(reading.count for reading in readings),
        _total(reading.memory_bytes for reading in readings),
        ", ".join(problems) or None,
    )
