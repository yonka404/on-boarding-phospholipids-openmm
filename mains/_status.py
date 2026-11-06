"""Read-only simulation status; standard library only, Python 3.8+."""

from __future__ import annotations

import argparse
import csv
import getpass
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from mains._common import system_id_argument
from mains._console import clean_text, paint

# These mirror the public profiles without importing the simulation package.
STAGES = {
    "solution": ("step4_equilibration", "step5_production"),
    "membrane": tuple(f"step6.{i}_equilibration" for i in range(1, 7)) + ("step7_production",),
}
_ACTIVE = {"RUNNING", "QUEUED", "STARTING", "FINISHING", "SUSPENDED", "RESIZING", "REQUEUED"}
_STATES = {
    "PENDING": "QUEUED", "CONFIGURING": "STARTING", "COMPLETING": "FINISHING",
    "STOPPED": "SUSPENDED", "REQUEUE_HOLD": "QUEUED", "SPECIAL_EXIT": "QUEUED",
    "STAGE_OUT": "FINISHING", "REQUEUE_FED": "QUEUED",
}
_READ_LIMIT = 65536


@dataclass(frozen=True)
class Execution:
    kind: str
    system_id: str
    identifier: str
    source: str
    state: str
    stage: str | None = None
    started_at: float | None = None
    node: str = ""
    elapsed: str = ""
    remaining: str = ""
    reason: str = ""

    @property
    def status(self) -> str:
        return _STATES.get(self.state, self.state)

    @property
    def active(self) -> bool:
        return self.status in _ACTIVE


@dataclass(frozen=True)
class Activity:
    executions: tuple[Execution, ...] = ()
    slurm_present: bool = False
    slurm_error: str | None = None
    local_error: str | None = None


@dataclass(frozen=True)
class Report:
    step: int | None = None
    speed: float | None = None
    modified: float | None = None
    problem: str | None = None
    unreadable: bool = False
    speed_step: int | None = None


@dataclass(frozen=True)
class StageProgress:
    name: str
    exists: bool = False
    complete: bool = False
    checkpoint: bool = False
    steps: int | None = None
    target_steps: int | None = None
    timestep_ps: float | None = None
    speed: float | None = None
    reported_at: float | None = None
    notes: tuple[str, ...] = ()
    unreadable: bool = False
    speed_step: int | None = None

    @property
    def percent(self) -> float | None:
        if self.complete:
            return 100.0
        if self.steps is None or self.target_steps is None:
            return None
        return self.steps / self.target_steps * 100

    @property
    def simulated_ns(self) -> float | None:
        return None if self.steps is None or self.timestep_ps is None else self.steps * self.timestep_ps / 1000

    @property
    def target_ns(self) -> float | None:
        return None if self.target_steps is None or self.timestep_ps is None else self.target_steps * self.timestep_ps / 1000


@dataclass(frozen=True)
class SystemStatus:
    kind: str
    system_id: str
    state: str
    stage_index: int
    stages: tuple[StageProgress, ...]
    executions: tuple[Execution, ...] = ()
    notes: tuple[str, ...] = ()
    log_path: Path | None = None

    @property
    def name(self) -> str:
        return f"{self.kind}/{self.system_id}"

    @property
    def current_stage(self) -> StageProgress:
        return self.stages[self.stage_index]

    @property
    def eta_seconds(self) -> float | None:
        stage = self.current_stage
        if self.state != "RUNNING" or sum(job.active for job in self.executions) != 1 or not stage.speed:
            return None
        if stage.simulated_ns is None or stage.target_ns is None:
            return None
        remaining = stage.target_ns - stage.simulated_ns
        estimate = max(0.0, remaining) / stage.speed * 86400
        return estimate if math.isfinite(estimate) else None


@dataclass(frozen=True)
class Snapshot:
    timestamp: float
    systems: tuple[SystemStatus, ...]
    notices: tuple[str, ...] = ()


def _csv_record(line: bytes, columns: list[str]) -> tuple[int, float | None, bool] | None:
    try:
        row = next(csv.reader([line.decode("utf-8")], strict=True))
        if len(row) != len(columns):
            return None
        step = int(row[columns.index("Step")])
        if step < 0:
            return None
        for index, value in enumerate(row):
            if columns[index] not in {"Step", "Speed (ns/day)"} and not math.isfinite(float(value)):
                return None
        speed_value = row[columns.index("Speed (ns/day)")].strip()
        if speed_value == "--":
            return step, None, True  # OpenMM's first report after reporter startup.
        speed = float(speed_value)
        return step, speed if math.isfinite(speed) and speed > 0 else None, False
    except (ValueError, UnicodeError, csv.Error, StopIteration):
        return None


def read_report(path: Path) -> Report:
    """Read a bounded header/tail from one file identity; retry one replacement."""
    for attempt in range(2):
        try:
            if path.is_symlink() or (path.exists() and not path.is_file()):
                return Report(problem="Simulation report is not a regular file; inspect outputs", unreadable=True)
            with path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                header = stream.readline(_READ_LIMIT)
                if not header.endswith(b"\n"):
                    return Report(modified=stat.st_mtime, problem="No complete CSV header")
                columns = next(csv.reader([header.decode("utf-8").lstrip("#")], strict=True))
                if not {"Step", "Speed (ns/day)"}.issubset(columns) or len(set(columns)) != len(columns):
                    return Report(modified=stat.st_mtime, problem="Unrecognized CSV header", unreadable=True)
                start = max(len(header), stat.st_size - _READ_LIMIT)
                stream.seek(start)
                tail = stream.read(_READ_LIMIT)
                if start > len(header):
                    tail = tail.partition(b"\n")[2]
                current = path.stat()
                if (stat.st_ino, stat.st_dev) != (current.st_ino, current.st_dev):
                    if attempt == 0:
                        continue
                    return Report(problem="Report replaced during snapshot; run status again")
                latest_step = None
                speed = None
                speed_step = None
                for line in reversed(tail.split(b"\n")[:-1]):
                    parsed = _csv_record(line.rstrip(b"\r"), columns)
                    if parsed is None:
                        continue
                    step, candidate_speed, startup = parsed
                    if latest_step is None:
                        latest_step = step
                    if speed is None and step <= latest_step and candidate_speed is not None:
                        speed = candidate_speed
                        speed_step = step
                    if speed is not None or startup:
                        break
                problem = "No complete valid simulation record in the report tail" if latest_step is None else None
                return Report(latest_step, speed, stat.st_mtime, problem, speed_step=speed_step)
        except FileNotFoundError:
            if attempt == 0 and path.parent.is_dir():
                continue
            return Report(problem="No simulation report yet")
        except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
            return Report(problem=f"Cannot read simulation report: {exc}", unreadable=True)
    return Report(problem="Report unavailable")


def _start_time(value: str) -> float | None:
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def _script_arguments(argv: list[str], cwd: Path, script: Path) -> list[str] | None:
    index = 1
    while index < len(argv) and argv[index].startswith("-"):
        if argv[index] in {"-c", "-m", "--help", "--version"}:
            return None
        if argv[index] in {"-W", "-X"}:
            index += 1
        index += 1
    if index >= len(argv) or (cwd / argv[index]).resolve() != script:
        return None
    return argv[index + 1:]


class _ProcessArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _local_arguments(argv: list[str], cwd: Path, root: Path) -> tuple[str, str, str | None] | None:
    args = _script_arguments(argv, cwd, root / "mains/local.py")
    if args is None:
        return None
    parser = _ProcessArgumentParser(add_help=False)
    parser.add_argument("kind", choices=STAGES)
    parser.add_argument("system_id", type=system_id_argument)
    parser.add_argument("--stage")
    try:
        parsed = parser.parse_args(args)
    except ValueError:
        return None
    if parsed.stage is not None and parsed.stage not in STAGES[parsed.kind]:
        return None
    return parsed.kind, parsed.system_id, parsed.stage


def _local_processes(root: Path, proc: Path, now: float) -> tuple[list[Execution], str | None]:
    executions = []
    try:
        uptime = float((proc / "uptime").read_text().split()[0])
        ticks = os.sysconf("SC_CLK_TCK")
        owner = os.getuid()
        directories = list(proc.iterdir())
    except (OSError, ValueError, AttributeError, IndexError) as exc:
        return [], f"Local process discovery unavailable: {exc}"
    problem = None
    for directory in directories:
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != owner:
                continue
            with (directory / "cmdline").open("rb") as stream:
                argv = [os.fsdecode(arg) for arg in stream.read(_READ_LIMIT).split(b"\0") if arg]
            if not argv or not Path(argv[0]).name.lower().startswith(("python", "pypy")):
                continue
            executable = (directory / "exe").resolve(strict=True).name.lower()
            if not executable.startswith(("python", "pypy")):
                continue
            before = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if before[0] in {"Z", "X"}:
                continue
            arguments = _local_arguments(argv, (directory / "cwd").resolve(strict=True), root)
            after = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if arguments is None or before[19] != after[19]:
                continue
            kind, system_id, stage = arguments
            source = "local"
            try:
                parent = proc / after[1]
                with (parent / "cmdline").open("rb") as stream:
                    parent_argv = [os.fsdecode(arg) for arg in stream.read(_READ_LIMIT).split(b"\0") if arg]
                if _script_arguments(parent_argv, (parent / "cwd").resolve(strict=True), root / "mains/_simcalc/run_solution.py") is not None:
                    source = "slurm-child"
            except OSError:
                pass  # Parent may have exited; never hide a possibly separate run.
            executions.append(Execution(
                kind, system_id, f"PID {directory.name}", source,
                "SUSPENDED" if after[0] in {"T", "t"} else "RUNNING",
                stage=stage, started_at=now - uptime + int(after[19]) / ticks,
            ))
        except (FileNotFoundError, ProcessLookupError):
            continue  # Processes routinely exit while /proc is being read.
        except (OSError, ValueError, IndexError):
            problem = "Some local process information is unreadable; activity may be unknown"
    return executions, problem


def read_activity(project_root: Path, *, proc_root: Path = Path("/proc"), now: float | None = None) -> Activity:
    root = project_root.resolve()
    local, local_error = _local_processes(root, proc_root, time.time() if now is None else now)
    slurm_present = shutil.which("squeue") is not None
    jobs = []
    slurm_error = None
    if slurm_present:
        try:
            try:
                import pwd
                user = pwd.getpwuid(os.getuid()).pw_name
            except (ImportError, AttributeError):
                user = getpass.getuser()
            result = subprocess.run(
                ["squeue", "--local", "-h", "-u", user, "-t", "all", "-o", "%i|%j|%T|%Z|%N|%M|%L|%S|%R"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode:
                slurm_error = f"Slurm unavailable: {result.stderr.strip() or 'squeue failed'}"
            else:
                for line in result.stdout.splitlines():
                    if not line.strip():
                        continue
                    fields = [part.strip() for part in line.split("|", 8)]
                    if len(fields) != 9:
                        slurm_error = "Some Slurm rows are unreadable; scheduler information is incomplete"
                        continue
                    job_id, name, state, cwd, node, elapsed, remaining, start, reason = fields
                    if not name.startswith("solution-md-") or not cwd or Path(cwd).resolve() != root:
                        continue
                    system_id = name[len("solution-md-"):]
                    if not re.fullmatch(r"[A-Za-z0-9_.-]+", system_id) or system_id in {".", ".."}:
                        continue
                    jobs.append(Execution(
                        "solution", system_id, job_id, "slurm", state,
                        started_at=_start_time(start), node=node, elapsed=elapsed,
                        remaining=remaining, reason=reason if reason not in {"None", "(null)", "N/A"} else "",
                    ))
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
            slurm_error = f"Slurm unavailable: {exc}"
    # On a compute node, local.py is the Slurm runner's child, not another run.
    scheduled = {(job.kind, job.system_id) for job in jobs if job.active}
    local = [job for job in local if job.source != "slurm-child" or (job.kind, job.system_id) not in scheduled]
    return Activity(tuple(jobs + local), slurm_present, slurm_error, local_error)


def _usable(path: Path) -> bool:
    return not path.is_symlink() and path.is_file() and path.stat().st_size > 0 and os.access(path, os.R_OK)


def _stage_progress(root: Path, kind: str, system_id: str, name: str) -> StageProgress:
    stage = root / "data/outputs/openmm_native" / kind / system_id / name
    protocol = root / "data/inputs/openmm_native" / kind / system_id / "openmm" / f"{name}.inp"
    notes = []
    target = timestep = None
    try:
        values = {}
        for raw in protocol.read_text().splitlines():
            line = raw.split("#", 1)[0].split("!", 1)[0]
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip().lower()] = value.strip()
        target, timestep = int(values["nstep"]), float(values["dt"])
        if target <= 0 or not math.isfinite(timestep) or timestep <= 0 or not math.isfinite(target * timestep):
            raise ValueError("nstep and dt must be finite and positive")
    except (OSError, UnicodeError, KeyError, ValueError, OverflowError) as exc:
        target = timestep = None
        notes.append(f"Stage targets unavailable ({protocol.name}): {exc}")
    try:
        exists = stage.exists() or stage.is_symlink()
        if exists and (stage.is_symlink() or not stage.is_dir()):
            return StageProgress(name, exists=True, notes=("Stage output is not a regular directory; inspect outputs",), unreadable=True)
        checkpoint = (stage / "checkpoint.chk").exists() or (stage / "checkpoint.chk").is_symlink()
        complete = exists and not checkpoint and all(_usable(stage / file) for file in ("final_state.xml", "final_coordinates.pdb"))
        report = read_report(stage / "state_data.csv") if exists else Report(problem="No simulation report yet")
        if not complete and report.problem:
            notes.append(report.problem)
        if report.step is not None and target is not None and report.step > target:
            notes.append("Reported steps exceed the input target; inspect inputs and outputs")
            report = replace(report, step=None, speed=None)
        if exists and not complete and not checkpoint:
            notes.append("Incomplete stage has no checkpoint; inspect outputs before retrying")
        return StageProgress(name, exists, complete, checkpoint, report.step, target, timestep, report.speed, report.modified, tuple(notes), report.unreadable, speed_step=report.speed_step)
    except OSError as exc:
        return StageProgress(name, exists=True, notes=(f"Cannot read stage outputs: {exc}",), unreadable=True)


def _retained_recovery_speed(stage_dir: Path, started: float | None, speed_step: int | None) -> str | None:
    """A repaired CSV has a new mtime even though its retained rows are old.

    Recovery preserves an original CSV in a newly created recovery-* directory.
    Until a speed-bearing step exceeds that original tail, attribution to the
    new execution is uncertain. Both current and backup CSV reads stay bounded.
    """
    try:
        backups = []
        for directory in stage_dir.iterdir():
            if not directory.name.startswith("recovery-") or directory.is_symlink() or not directory.is_dir():
                continue
            modified = directory.stat().st_mtime
            if started is not None and modified < started:
                continue
            saved = directory / "state_data.csv"
            if saved.exists() or saved.is_symlink():
                backups.append((modified, str(directory), saved))
        if backups:
            original = read_report(max(backups)[2])
            if original.step is None or speed_step is None or speed_step <= original.step:
                return "Retained recovery progress; speed/ETA wait for measurements beyond the pre-recovery report"
    except OSError:
        return "Recovery history unreadable; current speed/ETA cannot be confirmed"
    return None


def collect_snapshot(project_root: Path, *, now: float | None = None, activity: Activity | None = None) -> Snapshot:
    root = project_root.resolve()
    timestamp = time.time() if now is None else now
    if activity is None:
        activity = read_activity(root, now=timestamp)
    systems = {(job.kind, job.system_id) for job in activity.executions}
    notices = [error for error in (activity.slurm_error, activity.local_error) if error]
    logs = {}
    for kind in STAGES:
        directory = root / "data/outputs/openmm_native" / kind
        try:
            systems.update((kind, path.name) for path in directory.iterdir() if not path.name.startswith("."))
        except FileNotFoundError:
            pass
        except OSError as exc:
            notices.append(f"Cannot discover {kind} outputs: {exc}")
    try:
        for path in (root / "mains/logs").iterdir():
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)-(\d+)\.log", path.name)
            if match and path.is_file() and match[1] not in {".", ".."}:
                key = ("solution", match[1])
                systems.add(key)
                if key not in logs or int(match[2]) > logs[key][0]:
                    logs[key] = (int(match[2]), path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        notices.append(f"Cannot discover job logs: {exc}")
    results = []
    for kind, system_id in sorted(systems):
        stages = tuple(_stage_progress(root, kind, system_id, stage) for stage in STAGES[kind])
        executions = tuple(job for job in activity.executions if (job.kind, job.system_id) == (kind, system_id))
        active = tuple(job for job in executions if job.active)
        index = next((i for i, stage in enumerate(stages) if not stage.complete), len(stages) - 1)
        selected = next((job.stage for job in active if job.stage in STAGES[kind]), None)
        if selected:
            index = STAGES[kind].index(selected)
        notes = []
        output = root / "data/outputs/openmm_native" / kind / system_id
        invalid_output = False
        try:
            if output.is_symlink() or (output.exists() and not output.is_dir()):
                invalid_output = True
                notes.append("System output is not a regular directory; inspect outputs")
            elif output.exists():
                extra = {p.name for p in output.iterdir()} - set(STAGES[kind])
                if extra:
                    notes.append("Unrecognized output entries; inspect outputs: " + ", ".join(sorted(extra)))
            if not selected and any(stage.exists for stage in stages[index + 1:]) and not stages[index].complete:
                notes.append("Later-stage output exists after an incomplete stage; inspect stage order")
        except OSError as exc:
            notes.append(f"Cannot read system outputs: {exc}")
        if active:
            state = min(active, key=lambda job: ["RUNNING", "STARTING", "QUEUED", "FINISHING", "SUSPENDED", "RESIZING", "REQUEUED"].index(job.status)).status
            started = max((job.started_at for job in active if job.started_at is not None and job.status == "RUNNING"), default=None)
            stage = stages[index]
            if selected:
                stage = replace(stage, complete=False)
            if started is not None and stage.reported_at is not None and stage.reported_at < started:
                stage = replace(stage, complete=False, steps=None, speed=None, speed_step=None, reported_at=None, notes=stage.notes + ("Existing report predates this execution; waiting for a new simulation report",))
            elif selected and stage.steps is None:
                stage = replace(stage, notes=stage.notes + ("No simulation report yet for this single-stage execution",))
            if stage.speed is not None:
                retained = _retained_recovery_speed(output / stage.name, started, stage.speed_step)
                if retained:
                    stage = replace(stage, speed=None, speed_step=None, notes=stage.notes + (retained,))
            stages = stages[:index] + (stage,) + stages[index + 1:]
            if len(active) > 1:
                notes.append("Multiple active executions share these outputs; measurements cannot be attributed to one run")
        elif all(stage.complete for stage in stages) and not notes:
            state = "COMPLETE"
        elif any(stage.unreadable for stage in stages) or invalid_output:
            state = "UNREADABLE"
        elif activity.local_error or (kind == "solution" and (activity.slurm_error or ((kind, system_id) in logs and not activity.slurm_present))):
            state = "UNKNOWN"
            notes.append("Activity unknown; scheduler or local process information is unavailable")
        else:
            state = "UNFINISHED"
            notes.append("No active process or matching Slurm job found")
        for job in executions:
            if not job.active:
                notes.append(f"Slurm last reports {job.state} for job {job.identifier}")
        notes.extend(stages[index].notes if state != "COMPLETE" else ())
        log_path = next((root / "mains/logs" / f"{system_id}-{job.identifier}.log" for job in active if job.source == "slurm"), None)
        if log_path is None and (kind, system_id) in logs:
            log_path = logs[(kind, system_id)][1]
        results.append(SystemStatus(kind, system_id, state, index, stages, executions, tuple(dict.fromkeys(notes)), log_path))
    results.sort(key=lambda system: (0 if any(job.active for job in system.executions) else 2 if system.state == "COMPLETE" else 1, system.kind, system.system_id))
    return Snapshot(timestamp, tuple(results), tuple(notices))


def _duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "-"
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02}h"
    if hours:
        return f"{hours}h {minutes:02}m"
    return f"{minutes}m {seconds:02}s" if minutes else f"{seconds}s"


def _cell(value: object) -> str:
    return clean_text(value).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _state_tone(state: str) -> str:
    return "green" if state in {"RUNNING", "COMPLETE"} else "cyan" if state in {"QUEUED", "STARTING", "FINISHING"} else "yellow"


def render_snapshot(snapshot: Snapshot, *, width: int = 120, color: bool = False) -> str:
    lines = [paint("SIMULATIONS", "cyan", color) + " · snapshot at " + datetime.fromtimestamp(snapshot.timestamp).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")]
    if not snapshot.systems:
        lines.extend(["No simulations found in this checkout.", *(_cell(note) for note in snapshot.notices)])
        return "\n".join(lines) + "\n"
    counts = {}
    for system in snapshot.systems:
        label = "completed" if system.state == "COMPLETE" else system.state.lower()
        counts[label] = counts.get(label, 0) + 1
    lines.extend([" · ".join(f"{count} {state}" for state, count in counts.items()), ""])
    headers = ["System", "Job/PID", "State", "Stage", "Stage %", "ns/day", "Stage ETA", "Report age"]
    rows = []
    for system in snapshot.systems:
        stage = system.current_stage
        label = "Production" if stage.name.endswith("production") else "Equilibration"
        identifiers = ", ".join(job.identifier for job in system.executions if job.active) or "-"
        speed = stage.speed if system.state == "RUNNING" else None
        rows.append(list(map(_cell, [
            system.name, identifiers, system.state, f"{system.stage_index + 1}/{len(system.stages)} {label}",
            "-" if stage.percent is None else f"{stage.percent:.1f}%",
            "-" if speed is None else f"{speed:.3g}",
            "-" if system.eta_seconds is None else "~" + _duration(system.eta_seconds),
            _duration(snapshot.timestamp - stage.reported_at) if stage.reported_at is not None and stage.steps is not None else "-",
        ])))
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    if sum(widths) + 2 * (len(headers) - 1) <= width:
        def table_row(row: list[str], state: str | None = None) -> str:
            cells = [value.rjust(widths[i]) if i in {4, 5} else value.ljust(widths[i]) for i, value in enumerate(row)]
            if state:
                cells[2] = paint(cells[2], _state_tone(state), color)
            return "  ".join(cells).rstrip()
        lines.append(table_row(headers))
        lines.extend(table_row(row, system.state) for row, system in zip(rows, snapshot.systems))
    else:
        for row, system in zip(rows, snapshot.systems):
            lines.append(paint(row[0], "bold", color))
            for header, value in zip(headers[1:], row[1:]):
                lines.append(f"  {header:<11} " + (paint(value, _state_tone(system.state), color) if header == "State" else value))
            lines.append("")
    lines.extend(["", "Progress is last reported for the shown stage; ETA covers that stage only."])
    for notice in snapshot.notices:
        lines.append(paint("Notice: " + _cell(notice), "yellow", color))
    detailed = [system for system in snapshot.systems if system.state != "COMPLETE"]
    if detailed:
        lines.extend(["", paint("DETAILS", "cyan", color)])
    for system in detailed:
        stage = system.current_stage
        lines.append(_cell(system.name))
        progress = stage.name
        if stage.simulated_ns is not None and stage.target_ns is not None:
            progress += f" · {stage.simulated_ns:,.6g} / {stage.target_ns:,.6g} ns"
        if stage.steps is not None and stage.target_steps is not None:
            progress += f" · {stage.steps:,} / {stage.target_steps:,} steps"
        lines.append("  " + progress)
        lines.append(f"  Stages complete: {sum(stage.complete for stage in system.stages)}/{len(system.stages)}")
        for job in system.executions:
            if job.source == "slurm" and job.active:
                timing = [f"Job {job.identifier}"]
                if job.node and job.node not in {"(null)", "None", "N/A"}:
                    timing.append(f"node {job.node}")
                if job.status != "QUEUED":
                    timing.extend(f"{label} {value}" for label, value in (("elapsed", job.elapsed), ("allocation left", job.remaining)) if value)
                lines.append("  " + _cell(" · ".join(timing)))
                if job.reason:
                    lines.append("  Queue/reason: " + _cell(job.reason))
        for note in system.notes:
            lines.append("  " + _cell(note))
        if stage.checkpoint:
            lines.append("  Checkpoint present; recovery still requires runtime validation.")
        if stage.steps is not None and stage.speed is None and system.state == "RUNNING":
            lines.append("  Speed/ETA unavailable: no finite positive speed in the report tail.")
        if system.log_path:
            lines.append("  Log: " + _cell(system.log_path))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
