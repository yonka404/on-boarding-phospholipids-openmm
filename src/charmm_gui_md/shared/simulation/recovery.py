"""Validate and reconcile this project's OpenMM CSV/DCD checkpoint prefixes.

Only the retained prefix is authoritative. A crash can leave arbitrary bytes after
it, or a DCD frame count written before the corresponding frame reached disk.
"""

import csv
import logging
import math
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from charmm_gui_md.shared.artifacts import StageArtifacts
from charmm_gui_md.shared.protocols import OpenMMStageProtocol

logger = logging.getLogger(__name__)
_CHUNK = 64 * 1024
_CSV_HEADERS = [
    "Step",
    "Time (ps)",
    "Potential Energy (kJ/mole)",
    "Kinetic Energy (kJ/mole)",
    "Total Energy (kJ/mole)",
    "Temperature (K)",
    "Box Volume (nm^3)",
    "Density (g/mL)",
    "Speed (ns/day)",
]


@dataclass(frozen=True)
class _Prefix:
    path: Path
    length: int
    patches: tuple[tuple[int, bytes], ...] = ()

    @property
    def needs_repair(self) -> bool:
        if self.path.stat().st_size != self.length:
            return True
        with self.path.open("rb") as handle:
            for offset, data in self.patches:
                handle.seek(offset)
                if handle.read(len(data)) != data:
                    return True
        return False


def _csv_prefix(path: Path, step: int, interval: int) -> _Prefix:
    def row(handle: BinaryIO, *, header: bool = False) -> list[str]:
        line = handle.readline(_CHUNK)
        if not line.endswith(b"\n"):
            raise ValueError("missing or incomplete required CSV record")
        if header:
            if not line.startswith(b"#"):
                raise ValueError("missing CSV header")
            line = line[1:]
        return next(csv.reader([line.decode("utf-8")], strict=True))

    with path.open("rb") as handle:
        if row(handle, header=True) != _CSV_HEADERS:
            raise ValueError("CSV columns do not match the configured state reporter")
        for expected in range(interval, step + 1, interval):
            values = row(handle)
            if len(values) != len(_CSV_HEADERS) or int(values[0]) != expected:
                raise ValueError(
                    f"missing or out-of-order CSV record at step {expected}"
                )
            for index, value in enumerate(values[1:], 1):
                # OpenMM reports an unknown speed for the first record after startup.
                if index == 8 and value == "--":
                    continue
                if not math.isfinite(float(value)):
                    raise ValueError(f"nonfinite CSV value at step {expected}")
        return _Prefix(path, handle.tell())


def _read(handle: BinaryIO, length: int) -> bytes:
    data = handle.read(length)
    if len(data) != length:
        raise ValueError("missing or incomplete required DCD data")
    return data


def _integer(handle: BinaryIO) -> int:
    return struct.unpack("<i", _read(handle, 4))[0]


def _record(handle: BinaryIO, size: int, value_format: str) -> None:
    if _integer(handle) != size:
        raise ValueError("invalid DCD record size")
    remaining = size
    while remaining:
        data = _read(handle, min(remaining, _CHUNK))
        if any(
            not math.isfinite(value)
            for (value,) in struct.iter_unpack(value_format, data)
        ):
            raise ValueError("nonfinite value in required DCD frame")
        remaining -= len(data)
    if _integer(handle) != size:
        raise ValueError("invalid DCD record end marker")


def _dcd_prefix(
    path: Path,
    step: int,
    interval: int,
    timestep_ps: float,
    atoms: int,
    periodic: bool,
) -> _Prefix:
    retained = step // interval
    with path.open("rb") as handle:
        header = _read(handle, 92)
        integers = struct.unpack("<23i", header)
        if (
            integers[0] != 84
            or header[4:8] != b"CORD"
            or any(integers[6:11])
            or any(integers[13:21])
            or integers[21:23] != (24, 84)
            or integers[12] != int(periodic)
        ):
            raise ValueError("unsupported or corrupt OpenMM DCD header")
        count, first, stride = integers[2:5]
        delta = struct.unpack_from("<f", header, 44)[0]
        # OpenMM rescales the header when trajectory step indices overflow int32.
        scale = interval if stride == 1 and interval != 1 else 1
        if (
            first != stride
            or stride * scale != interval
            or not math.isclose(delta, timestep_ps * scale / 0.04888821, rel_tol=1e-6)
            or count < retained
        ):
            raise ValueError(
                "DCD frame count, interval or timestep does not match checkpoint/protocol"
            )
        title_size = _integer(handle)
        if title_size != 164 or _integer(handle) != 2:
            raise ValueError("unsupported OpenMM DCD title block")
        _read(handle, 160)
        if _integer(handle) != title_size or _integer(handle) != 4:
            raise ValueError("invalid DCD title/atom record")
        if _integer(handle) != atoms or atoms <= 0 or _integer(handle) != 4:
            raise ValueError("DCD atom count does not match the simulation")
        for _ in range(retained):
            if periodic:
                _record(handle, 48, "<d")
            for _ in range(3):
                _record(handle, 4 * atoms, "<f")
        # DCDFile(append=True) restores only the count, not the timing fields.
        # Match the header OpenMM would have written for this retained prefix;
        # a long discarded tail may have triggered scaling too early for it.
        output_scale = (
            interval if interval > 1 and (retained + 1) * interval > 1 << 31 else 1
        )
        output_stride = interval // output_scale
        return _Prefix(
            path,
            handle.tell(),
            (
                (
                    8,
                    struct.pack(
                        "<4i",
                        retained,
                        output_stride,
                        output_stride,
                        retained * output_stride,
                    ),
                ),
                (44, struct.pack("<f", timestep_ps / 0.04888821 * output_scale)),
            ),
        )


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replacement(prefix: _Prefix, destination: Path) -> None:
    with prefix.path.open("rb") as source, destination.open("xb") as target:
        remaining = prefix.length
        while remaining:
            block = _read(source, min(remaining, _CHUNK))
            target.write(block)
            remaining -= len(block)
        for offset, data in prefix.patches:
            target.seek(offset)
            target.write(data)
        target.flush()
        os.fsync(target.fileno())
    shutil.copymode(prefix.path, destination)


def reconcile_reporters(
    artifacts: StageArtifacts,
    protocol: OpenMMStageProtocol,
    *,
    checkpoint_step: int,
    atom_count: int,
    periodic: bool,
) -> None:
    """Finish validation and backups before atomically replacing affected files.

    The checkpoint is never changed here. If only one replacement completes,
    reloading that same checkpoint and repeating reconciliation is safe.
    """
    if not 0 <= checkpoint_step <= protocol.dynamics_steps:
        raise ValueError(f"Invalid checkpoint step: {checkpoint_step}")
    if (
        min(
            protocol.state_report_interval_steps,
            protocol.trajectory_report_interval_steps,
        )
        <= 0
    ):
        raise ValueError("Reporter intervals must be positive for recovery")
    for path in (artifacts.state_data_path, artifacts.trajectory_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Cannot recover reporter {path}: expected a regular file")
    prefixes = []
    for path, validate in (
        (
            artifacts.state_data_path,
            lambda: _csv_prefix(
                artifacts.state_data_path,
                checkpoint_step,
                protocol.state_report_interval_steps,
            ),
        ),
        (
            artifacts.trajectory_path,
            lambda: _dcd_prefix(
                artifacts.trajectory_path,
                checkpoint_step,
                protocol.trajectory_report_interval_steps,
                protocol.timestep_ps,
                atom_count,
                periodic,
            ),
        ),
    ):
        try:
            prefixes.append(validate())
        except (ValueError, UnicodeError, csv.Error, struct.error) as exc:
            raise ValueError(
                f"Cannot recover {path}: {exc}. Inspect the retained prefix; outputs were left unchanged."
            ) from exc
    affected = [prefix for prefix in prefixes if prefix.needs_repair]
    if not affected:
        return

    backup = Path(tempfile.mkdtemp(prefix="recovery-", dir=artifacts.output_dir))
    logger.info(
        "[%s] Preparing reporter recovery backups in %s", protocol.step_name, backup
    )
    # All backups must be durable before *either* original is replaced.
    for prefix in affected:
        saved = backup / prefix.path.name
        shutil.copy2(prefix.path, saved)
        with saved.open("rb") as handle:
            os.fsync(handle.fileno())
    _sync_directory(backup)
    _sync_directory(artifacts.output_dir)
    for prefix in affected:
        _replacement(prefix, backup / (prefix.path.name + ".replacement"))
    for prefix in affected:
        os.replace(backup / (prefix.path.name + ".replacement"), prefix.path)
        _sync_directory(artifacts.output_dir)
    logger.info(
        "[%s] Reporter files reconciled through checkpoint step %d; originals saved in %s",
        protocol.step_name,
        checkpoint_step,
        backup,
    )
