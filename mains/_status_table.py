"""Render a read-only simulation snapshot for wide and narrow terminals."""

from __future__ import annotations

import unicodedata
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

from mains._console import paint

if TYPE_CHECKING:
    from mains._status import Execution, Snapshot, StageProgress, SystemStatus


_MISSING = "—"
# The keys also serve as the full-width headers and labels in Details.
_COMPACT = {
    "State": "State",
    "Stage": "Stage",
    "Stage done": "Done",
    "GPUs used": "GPUs",
    "VRAM used": "VRAM",
    "ns/day": "ns/day",
    "Est. stage time left": "Stage time left",
    "Progress file updated": "File updated",
    "Job/PID": "Job/PID",
}
_NUMERIC = {"Stage done", "GPUs used", "VRAM used", "ns/day"}


def _cell(value: object) -> str:
    # Filenames and scheduler messages can contain terminal controls or newlines.
    return "".join(c if c.isprintable() else ascii(c)[1:-1] for c in str(value))


def _width(text: str) -> int:
    if text.isascii():
        return len(text)
    return sum(
        2 if unicodedata.east_asian_width(c) in {"W", "F"} else 1
        for c in text if not unicodedata.combining(c)
    )


def _prefix(text: str, limit: int) -> str:
    used = 0
    for index, char in enumerate(text):
        used += _width(char)
        if used > limit:
            return text[:index]
    return text


def _short_name(name: str, limit: int) -> str:
    if _width(name) <= limit:
        return name
    # Keep more of the suffix, where CHARMM-GUI IDs commonly distinguish replicas.
    head = max(2, (limit - 1) // 3)
    return _prefix(name, head) + "…" + _prefix(name[::-1], limit - 1 - head)[::-1]


def _names(systems: tuple[SystemStatus, ...], room: int) -> list[str]:
    labels = []
    for system in systems:
        full = _cell(system.name)
        labels.append(full if _width(full) <= room else _short_name(_cell(system.system_id), room))
    if len(set(labels)) == len(labels):
        return labels
    # Replicas can differ only in the middle, or share an ID across kinds.
    # Number all rows in this case so literal user IDs cannot collide with an alias.
    numbered = []
    for index, system in enumerate(systems, 1):
        prefix = f"#{index} "
        numbered.append(prefix + _short_name(_cell(system.system_id), room - len(prefix)))
    return numbered


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days} d {hours} hr" if hours else f"{days} d"
    if hours:
        return f"{hours} hr {minutes} min" if minutes else f"{hours} hr"
    return f"{minutes} min" if minutes else f"{seconds} sec"


def _memory(size: int | None) -> str:
    if size is None:
        return _MISSING
    if size == 0:
        return "0 MiB"
    if size < 1024**2:
        return "<1 MiB"
    amount, unit = (size / 1024**2, "MiB") if size < 1024**3 else (size / 1024**3, "GiB")
    return f"{amount:.0f} {unit}" if amount >= 100 else f"{amount:.3g} {unit}"


def _percent(value: float | None) -> str:
    if value is None:
        return _MISSING
    if 0 < value < 0.1:
        return "<0.1%"
    if 99.9 < value < 100:
        return ">99.9%"
    return f"{value:.1f}%"


def _values(system: SystemStatus, now: float) -> dict[str, str]:
    stage = system.current_stage
    stage_type = "Production" if stage.name.endswith("production") else "Equilibration"
    active = [job for job in system.executions if job.active]
    count = system.gpu.count
    eta = system.eta_seconds
    speed = system.speed
    age = None if stage.reported_at is None else now - stage.reported_at
    return {
        "State": _cell(system.state),
        "Stage": f"{system.stage_index + 1}/{len(system.stages)} {stage_type}",
        "Stage done": _percent(stage.percent),
        "GPUs used": _MISSING if count is None else f"{count}!" if count > 1 else str(count),
        "VRAM used": _memory(system.gpu.memory_bytes),
        "ns/day": _MISSING if speed is None else f"{speed:.3g}",
        "Est. stage time left": _MISSING if eta is None else "~" + _duration(eta),
        "Progress file updated": _MISSING if age is None else _duration(age) + " ago",
        "Job/PID": _cell(active[0].identifier) if len(active) == 1 else f"{len(active)} active" if active else _MISSING,
    }


def _widths(headers: dict[str, str], rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        key: max(_width(header), max(_width(row[key]) for row in rows))
        for key, header in headers.items()
    }


def _layout(systems: tuple[SystemStatus, ...], rows: list[dict[str, str]], width: int):
    headers = {key: key for key in _COMPACT}
    widths = _widths(headers, rows)
    # Every data column has a two-space gutter after the name or previous column.
    room = width - sum(widths.values()) - 2 * len(widths)
    desired = min(20, max(_width(_cell(system.name)) for system in systems))
    if room < desired:
        optional = ("Job/PID", "Progress file updated", "ns/day", "Est. stage time left")
        headers = {
            key: label for key, label in _COMPACT.items()
            if key not in optional or any(row[key] != _MISSING for row in rows)
        }
        for row in rows:
            row["Stage"] = row["Stage"].split()[0]
        widths = _widths(headers, rows)
        room = width - sum(widths.values()) - 2 * len(widths)
        for key in optional:
            if room >= desired:
                break
            if key in headers:
                room += widths.pop(key) + 2
                del headers[key]

    digits = len(str(len(rows)))
    if room >= max(8, digits + 5):
        labels = _names(systems, room)
        name_header = "System"
    else:
        # On a very small screen the row number retains an unambiguous link to Details.
        labels = [str(index) for index in range(1, len(rows) + 1)]
        name_header = "#"
        for key in ("Stage", "VRAM used", "GPUs used", "Stage done", "State"):
            if room >= digits:
                break
            room += widths.pop(key) + 2
            del headers[key]

    headers = {"System": name_header, **headers}
    name_width = max(len(name_header), max(_width(name) for name in labels))
    return headers, {"System": name_width, **widths}, labels


def _state_tone(state: str) -> str:
    if state in {"RUNNING", "COMPLETE"}:
        return "green"
    if state in {"QUEUED", "STARTING", "FINISHING"}:
        return "cyan"
    return "red" if state == "UNREADABLE" else "yellow"


def _table_line(values: dict[str, str], widths: dict[str, int], color: bool, *, header: bool = False) -> str:
    cells = []
    for key, size in widths.items():
        value = values[key]
        pad = " " * (size - _width(value))
        if header:
            value = paint(value, "bold", color)
        elif key == "State":
            value = paint(value, _state_tone(value), color)
        elif key == "GPUs used" and value.endswith("!"):
            value = paint(value, "yellow", color)
        cells.append(pad + value if key in _NUMERIC else value + pad)
    return "  ".join(cells).rstrip()


def _progress(stage: StageProgress) -> list[str]:
    parts = []
    for current, target, unit, fmt in (
        (stage.simulated_ns, stage.target_ns, "ns", ",.15g"),
        (stage.steps, stage.target_steps, "steps", ",d"),
    ):
        if current is None:
            if target is not None:
                parts.append(f"target {target:{fmt}} {unit}")
        elif target is None:
            parts.append(f"{current:{fmt}} {unit}")
        else:
            parts.append(f"{current:{fmt}} / {target:{fmt}} {unit}")
    return parts


def _execution_line(job: Execution, show_id: bool) -> str:
    parts = []
    if show_id:
        parts.append(("Job " if job.source == "slurm" else "") + job.identifier)
    if job.source == "slurm":
        if job.status == "QUEUED":
            if job.reason:
                parts.append("Queue reason " + job.reason)
        else:
            if job.node and job.node not in {"(null)", "None", "N/A"}:
                parts.append("Node " + job.node)
            if job.elapsed:
                parts.append("Elapsed " + job.elapsed)
            if job.remaining:
                parts.append("Allocation left " + job.remaining)
    return _cell(" · ".join(parts))


def _details(system: SystemStatus, row: dict[str, str], headers: dict[str, str], color: bool) -> list[str]:
    lines = []
    stage = system.current_stage
    omitted = [
        (f"{value} ns/day" if key == "ns/day" else f"{key} {value}")
        for key, value in row.items()
        if key not in headers and key not in {"Job/PID", "Stage"} and value != _MISSING
    ]
    if omitted:
        lines.append(" · ".join(omitted))
    if system.state != "COMPLETE" or "Stage" not in headers or " " not in row["Stage"]:
        stage_name = _cell(stage.name)
        if "Stage" not in headers:
            stage_name = f"{system.stage_index + 1}/{len(system.stages)} " + stage_name
        progress = [stage_name]
        if system.state != "COMPLETE":
            progress.extend(_progress(stage))
        completed = sum(part.complete for part in system.stages)
        if system.state != "COMPLETE" and completed != system.stage_index:
            progress.append(f"{completed}/{len(system.stages)} stages complete")
        lines.append(" · ".join(progress))
    active = [job for job in system.executions if job.active]
    for job in active:
        info = _execution_line(job, show_id="Job/PID" not in headers or len(active) > 1)
        if info:
            lines.append(info)
    if stage.checkpoint and system.state in {"QUEUED", "UNKNOWN", "UNREADABLE", "UNFINISHED"}:
        lines.append("Checkpoint present")
    for note in system.notes:
        if system.state == "RUNNING" and note.startswith("Incomplete stage has no checkpoint"):
            continue
        if note == "No simulation report yet" and any(other.startswith(note + " for") for other in system.notes):
            continue
        lines.append(paint(_cell(note), "yellow", color))
    if system.log_path:
        label = "Job log" if active else "Last job log"
        lines.append(label + " " + _cell(system.log_path))
    return lines


def render_snapshot(snapshot: Snapshot, *, width: int = 120, color: bool = False) -> str:
    time_label = datetime.fromtimestamp(snapshot.timestamp).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [paint("SIMULATIONS", "cyan", color) + " · snapshot at " + time_label]
    notices = [paint("Notice · " + _cell(note), "yellow", color) for note in snapshot.notices]
    if not snapshot.systems:
        return "\n".join([*lines, "No simulations found in this checkout.", *notices]) + "\n"

    systems = snapshot.systems
    counts = Counter("completed" if s.state == "COMPLETE" else s.state.lower() for s in systems)
    lines.extend([" · ".join(f"{count} {state}" for state, count in counts.items()), ""])
    rows = [_values(system, snapshot.timestamp) for system in systems]
    headers, widths, names = _layout(systems, rows, width)
    lines.append(_table_line(headers, widths, color, header=True))
    for name, row in zip(names, rows):
        lines.append(_table_line({"System": name, **row}, widths, color))

    legend = []
    if any(_MISSING in row.values() for row in rows):
        legend.append("— unavailable")
    if any(row["GPUs used"].endswith("!") for row in rows):
        legend.append("! using more than one GPU")
    if legend or notices:
        lines.append("")
        if legend:
            lines.append(" · ".join(legend))
        lines.extend(notices)

    blocks = []
    for system, name, row in zip(systems, names, rows):
        extra = _details(system, row, headers, color)
        problem = system.gpu.problem
        if problem:
            extra.append(paint("GPU data · " + _cell(problem), "yellow", color))
        full = _cell(system.name)
        if extra or name != full:
            title = full if full == name else f"{name}  →  {full}"
            blocks.append("\n".join([paint(title, "bold", color), *("  " + line for line in extra)]))
    if blocks:
        lines.extend(["", paint("DETAILS", "cyan", color), "\n\n".join(blocks)])
    return "\n".join(lines) + "\n"
