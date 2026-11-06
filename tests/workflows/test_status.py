"""Read-only status snapshots using temporary files and scheduler/OS fixtures."""

import ast
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mains import _status

PROJECT = Path(__file__).resolve().parents[2]
HEADER = (
    '#"Step","Time (ps)","Potential Energy (kJ/mole)","Kinetic Energy (kJ/mole)",'
    '"Total Energy (kJ/mole)","Temperature (K)","Box Volume (nm^3)",'
    '"Density (g/mL)","Speed (ns/day)"\n'
)
NOW = 1_700_000_000.0


def record(step, speed="10", simulation_time="1000000"):
    return f"{step},{simulation_time},-100,50,-50,300,100,1,{speed}\n"


class StatusFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project with spaces"
        self.root.mkdir()

    def stage(self, name="step5_production", kind="solution", system="replica_2", *, complete=False):
        inputs = self.root / "data/inputs/openmm_native" / kind / system / "openmm"
        inputs.mkdir(parents=True, exist_ok=True)
        (inputs / f"{name}.inp").write_text("nstep = 250000000 # steps\ndt = 0.002 ! ps\n")
        path = self.root / "data/outputs/openmm_native" / kind / system / name
        path.mkdir(parents=True, exist_ok=True)
        if complete:
            (path / "final_state.xml").write_text("<state/>")
            (path / "final_coordinates.pdb").write_text("ATOM")
        return path

    def csv(self, stage, content):
        path = stage / "state_data.csv"
        path.write_text(HEADER + content)
        os.utime(path, (NOW - 28, NOW - 28))
        return path

    def snapshot(self, *executions, **activity_options):
        return _status.collect_snapshot(
            self.root, now=NOW,
            activity=_status.Activity(executions=tuple(executions), **activity_options),
        )

    def job(self, state="RUNNING", **options):
        return _status.Execution(
            kind="solution", system_id="replica_2", identifier="321",
            source="slurm", state=state, **options,
        )


class StatusTests(StatusFixture):
    def test_running_stage_uses_steps_not_cumulative_csv_time(self):
        self.stage("step4_equilibration", complete=True)
        stage = self.stage()
        self.csv(stage, record(160_000_000, "13.4"))
        system = self.snapshot(self.job()).systems[0]
        self.assertEqual(system.state, "RUNNING")
        self.assertEqual(system.current_stage.steps, 160_000_000)
        self.assertEqual(system.current_stage.percent, 64.0)
        self.assertEqual(system.current_stage.simulated_ns, 320.0)
        self.assertEqual(system.current_stage.target_ns, 500.0)
        self.assertAlmostEqual(system.eta_seconds, 1_160_597.0149253731)

    def test_queued_fresh_job_is_discovered_before_outputs_exist(self):
        system = self.snapshot(self.job("PENDING", reason="Resources")).systems[0]
        self.assertEqual(system.state, "QUEUED")
        self.assertIsNone(system.current_stage.steps)
        self.assertIsNone(system.eta_seconds)
        output = _status.render_snapshot(self.snapshot(self.job("PENDING", reason="Resources")))
        self.assertIn("Resources", output)
        self.assertIn("No simulation report", output)

    def test_queued_resume_shows_last_progress_without_eta(self):
        stage = self.stage("step4_equilibration")
        self.csv(stage, record(125_000_000, "10"))
        (stage / "checkpoint.chk").write_bytes(b"checkpoint")
        system = self.snapshot(self.job("PENDING")).systems[0]
        self.assertEqual(system.current_stage.percent, 50)
        self.assertIsNone(system.eta_seconds)

    def test_complete_requires_all_stages_final_files_and_no_checkpoint(self):
        self.stage("step4_equilibration", complete=True)
        stage = self.stage(complete=True)
        self.assertEqual(self.snapshot().systems[0].state, "COMPLETE")
        (stage / "checkpoint.chk").write_bytes(b"checkpoint")
        self.assertEqual(self.snapshot().systems[0].state, "UNFINISHED")
        (stage / "checkpoint.chk").unlink()
        (stage / "final_coordinates.pdb").write_text("")
        self.assertNotEqual(self.snapshot().systems[0].state, "COMPLETE")

    def test_inactive_checkpoint_is_unfinished_not_failed_and_has_no_eta(self):
        stage = self.stage("step4_equilibration")
        (stage / "checkpoint.chk").write_bytes(b"checkpoint")
        self.csv(stage, record(1000))
        system = self.snapshot().systems[0]
        self.assertEqual(system.state, "UNFINISHED")
        self.assertIsNone(system.eta_seconds)
        self.assertIn("checkpoint present", _status.render_snapshot(self.snapshot()).lower())

    def test_missing_checkpoint_does_not_claim_recovery_is_possible(self):
        stage = self.stage("step4_equilibration")
        self.csv(stage, record(1000))
        output = _status.render_snapshot(self.snapshot())
        self.assertIn("checkpoint", output.lower())
        self.assertIn("inspect", output.lower())
        self.assertNotIn("FAILED", output)

    def test_unavailable_activity_preserves_progress_and_shows_unknown(self):
        stage = self.stage("step4_equilibration")
        self.csv(stage, record(1000))
        snapshot = self.snapshot(slurm_present=True, slurm_error="Slurm controller unavailable")
        self.assertEqual(snapshot.systems[0].state, "UNKNOWN")
        self.assertEqual(snapshot.systems[0].current_stage.steps, 1000)
        self.assertIn("Slurm controller unavailable", _status.render_snapshot(snapshot))
        self.assertIsNone(snapshot.systems[0].eta_seconds)

    def test_input_only_systems_and_hidden_locks_are_excluded(self):
        (self.root / "data/inputs/openmm_native/solution/unused/openmm").mkdir(parents=True)
        (self.root / "data/outputs/openmm_native/solution/.locked.lock").mkdir(parents=True)
        self.assertEqual(self.snapshot().systems, ())

    def test_job_log_names_preserve_system_ids_with_hyphens(self):
        logs = self.root / "mains/logs"
        logs.mkdir(parents=True)
        (logs / "abeta-42-replica-2-123.log").write_text("old log")
        (logs / "unrelated.log").write_text("ignored")
        snapshot = self.snapshot()
        self.assertEqual([s.system_id for s in snapshot.systems], ["abeta-42-replica-2"])
        self.assertEqual(snapshot.systems[0].state, "UNKNOWN")

    def test_single_stage_completion_is_not_whole_protocol_completion(self):
        self.stage("step4_equilibration", complete=True)
        system = self.snapshot().systems[0]
        self.assertEqual(system.state, "UNFINISHED")
        self.assertEqual(system.current_stage.name, "step5_production")

    def test_explicit_membrane_stage_is_selected_for_local_run(self):
        self.stage("step6.3_equilibration", kind="membrane", system="ligand")
        execution = _status.Execution(
            kind="membrane", system_id="ligand", identifier="PID 4128",
            source="local", state="RUNNING", stage="step6.3_equilibration",
        )
        system = self.snapshot(execution).systems[0]
        self.assertEqual(system.current_stage.name, "step6.3_equilibration")
        self.assertEqual(len(system.stages), 7)

    def test_new_execution_does_not_reuse_previous_execution_measurements(self):
        stage = self.stage("step4_equilibration", complete=True)
        self.csv(stage, record(250_000_000))
        execution = _status.Execution(
            kind="solution", system_id="replica_2", identifier="PID 42", source="local",
            state="RUNNING", stage="step4_equilibration", started_at=NOW - 10,
        )
        system = self.snapshot(execution).systems[0]
        self.assertEqual(system.state, "RUNNING")
        self.assertIsNone(system.current_stage.steps)
        self.assertIsNone(system.eta_seconds)
        self.assertIn("predates", _status.render_snapshot(self.snapshot(execution)))

    def test_progress_can_roll_back_when_recovery_replaces_csv(self):
        stage = self.stage("step4_equilibration")
        path = self.csv(stage, record(2000))
        self.assertEqual(self.snapshot(self.job()).systems[0].current_stage.steps, 2000)
        replacement = path.with_suffix(".replacement")
        replacement.write_text(HEADER + record(1000))
        replacement.replace(path)
        self.assertEqual(self.snapshot(self.job()).systems[0].current_stage.steps, 1000)

    def test_symlink_outputs_are_not_reported_as_complete(self):
        self.stage("step4_equilibration", complete=True)
        stage = self.stage(complete=True)
        state = stage / "final_state.xml"
        state.unlink()
        state.symlink_to(stage / "final_coordinates.pdb")
        self.assertNotEqual(self.snapshot().systems[0].state, "COMPLETE")

    def test_malformed_report_is_explicitly_unreadable_without_claiming_failure(self):
        stage = self.stage("step4_equilibration")
        (stage / "state_data.csv").write_text("unknown,header\n1,2\n")
        system = self.snapshot().systems[0]
        self.assertEqual(system.state, "UNREADABLE")

    def test_explicit_rerun_with_old_final_files_and_no_csv_waits_for_progress(self):
        self.stage("step4_equilibration", complete=True)
        execution = _status.Execution(
            "solution", "replica_2", "PID 42", "local", "RUNNING",
            stage="step4_equilibration", started_at=NOW - 10,
        )
        system = self.snapshot(execution).systems[0]
        self.assertIsNone(system.current_stage.percent)
        self.assertIn("No simulation report", _status.render_snapshot(self.snapshot(execution)))

    def test_invalid_targets_preserve_steps_but_suppress_percentage_and_eta(self):
        stage = self.stage("step4_equilibration")
        self.csv(stage, record(1000))
        protocol = self.root / "data/inputs/openmm_native/solution/replica_2/openmm/step4_equilibration.inp"
        for target, dt in (("0", "0.002"), ("1000", "nan"), ("1000", "0"), ("bad", "0.002")):
            with self.subTest(target=target, dt=dt):
                protocol.write_text(f"nstep = {target}\ndt = {dt}\n")
                system = self.snapshot(self.job()).systems[0]
                self.assertEqual(system.current_stage.steps, 1000)
                self.assertIsNone(system.current_stage.percent)
                self.assertIsNone(system.eta_seconds)

    def test_non_directory_output_is_visible_as_unreadable(self):
        root = self.root / "data/outputs/openmm_native/solution"
        root.mkdir(parents=True)
        (root / "broken").write_text("not a system directory")
        snapshot = self.snapshot()
        self.assertEqual([system.system_id for system in snapshot.systems], ["broken"])
        self.assertEqual(snapshot.systems[0].state, "UNREADABLE")

    def test_active_then_unfinished_then_complete_sorting(self):
        self.stage("step4_equilibration", system="a_done", complete=True)
        self.stage(system="a_done", complete=True)
        self.stage("step4_equilibration", system="b_unfinished")
        execution = _status.Execution("solution", "z_running", "42", "slurm", "RUNNING")
        self.assertEqual(
            [system.system_id for system in self.snapshot(execution).systems],
            ["z_running", "b_unfinished", "a_done"],
        )

    def test_monitor_schedules_match_public_workflows(self):
        from charmm_gui_md import membrane, solution
        self.assertEqual(_status.STAGES["solution"], solution.PROTOCOL_STAGES)
        self.assertEqual(_status.STAGES["membrane"], membrane.PROTOCOL_STAGES)

    def test_inactive_job_history_does_not_suppress_a_running_eta(self):
        self.stage("step4_equilibration", complete=True)
        stage = self.stage()
        self.csv(stage, record(160_000_000, "13.4"))
        old = _status.Execution("solution", "replica_2", "320", "slurm", "COMPLETED")
        system = self.snapshot(self.job(), old).systems[0]
        self.assertAlmostEqual(system.eta_seconds, 1_160_597.0149253731)

    def test_extreme_positive_speed_cannot_produce_nonfinite_eta(self):
        stage = self.stage("step4_equilibration")
        self.csv(stage, record(1000, "1e-320"))
        self.assertIsNone(self.snapshot(self.job()).systems[0].eta_seconds)

    def test_explicit_rerun_uses_new_steps_even_while_old_final_files_remain(self):
        stage = self.stage("step4_equilibration", complete=True)
        path = self.csv(stage, record(2_500_000, "10"))
        os.utime(path, (NOW - 5, NOW - 5))
        execution = _status.Execution(
            "solution", "replica_2", "PID 42", "local", "RUNNING",
            stage="step4_equilibration", started_at=NOW - 10,
        )
        system = self.snapshot(execution).systems[0]
        self.assertEqual(system.current_stage.percent, 1.0)
        self.assertEqual(sum(stage.complete for stage in system.stages), 0)

    def test_rewritten_recovery_prefix_does_not_supply_current_speed_or_eta(self):
        stage = self.stage("step4_equilibration")
        (stage / "checkpoint.chk").write_bytes(b"checkpoint")
        backup = stage / "recovery-example"
        backup.mkdir()
        (backup / "state_data.csv").write_text(HEADER + record(12_000, "8"))
        os.utime(backup, (NOW - 7, NOW - 7))
        path = self.csv(stage, record(10_000, "8"))
        os.utime(path, (NOW - 5, NOW - 5))
        execution = self.job(started_at=NOW - 10)
        system = self.snapshot(execution).systems[0]
        self.assertEqual(system.current_stage.steps, 10_000)
        self.assertIsNone(system.current_stage.speed)
        self.assertIsNone(system.eta_seconds)
        self.assertIn("retained", _status.render_snapshot(self.snapshot(execution)).lower())
        with path.open("a") as stream:
            stream.write(record(13_000, "12"))
        os.utime(path, (NOW - 1, NOW - 1))
        system = self.snapshot(execution).systems[0]
        self.assertEqual(system.current_stage.speed, 12)
        self.assertIsNotNone(system.eta_seconds)

    def test_old_recovery_backups_do_not_hide_new_execution_measurements(self):
        stage = self.stage("step4_equilibration")
        backup = stage / "recovery-old"
        backup.mkdir()
        (backup / "state_data.csv").write_text(HEADER + record(12_000, "8"))
        os.utime(backup, (NOW - 100, NOW - 100))
        path = self.csv(stage, record(10_000, "12"))
        os.utime(path, (NOW - 5, NOW - 5))
        self.assertIsNotNone(self.snapshot(self.job(started_at=NOW - 10)).systems[0].eta_seconds)


class CsvStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state_data.csv"

    def test_partial_malformed_and_noninteger_records_are_ignored(self):
        self.path.write_text(HEADER + record(1000) + "broken,row\n" + record("1.5") + "2000,3")
        report = _status.read_report(self.path)
        self.assertEqual(report.step, 1000)
        self.assertEqual(report.speed, 10)

    def test_last_finite_positive_speed_is_used(self):
        for speed in ("0", "nan", "inf", "-2"):
            with self.subTest(speed=speed):
                self.path.write_text(HEADER + record(1000, "8") + record(2000, speed))
                report = _status.read_report(self.path)
                self.assertEqual(report.step, 2000)
                self.assertEqual(report.speed, 8)

    def test_unusable_speed_does_not_remove_valid_step_progress(self):
        self.path.write_text(HEADER + record(1000, "nan"))
        report = _status.read_report(self.path)
        self.assertEqual(report.step, 1000)
        self.assertIsNone(report.speed)

    def test_unknown_startup_speed_preserves_steps_without_borrowing_old_run_speed(self):
        self.path.write_text(HEADER + record(1000, "8") + record(2000, "--"))
        report = _status.read_report(self.path)
        self.assertEqual(report.step, 2000)
        self.assertIsNone(report.speed)

    def test_missing_or_malformed_header_reports_problem(self):
        for contents in ("", "bad header\n" + record(1000), HEADER):
            with self.subTest(contents=contents):
                self.path.write_text(contents)
                report = _status.read_report(self.path)
                self.assertIsNone(report.step)
                self.assertTrue(report.problem)

    def test_large_csv_reads_are_bounded(self):
        with self.path.open("wb") as stream:
            stream.write(HEADER.encode())
            stream.seek(20_000_000)
            stream.write(b"\n" + record(1000).encode())
        original_open = Path.open
        bytes_read = []

        class Reader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def read(self, size=-1):
                if size < 0 or size > 65536:
                    raise AssertionError(f"unbounded read: {size}")
                value = self.stream.read(size)
                bytes_read.append(len(value))
                return value

            def readline(self, size=-1):
                if size < 0 or size > 65536:
                    raise AssertionError(f"unbounded readline: {size}")
                value = self.stream.readline(size)
                bytes_read.append(len(value))
                return value

        def bounded_open(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return Reader(stream) if path == self.path else stream

        with mock.patch.object(Path, "open", bounded_open):
            report = _status.read_report(self.path)
        self.assertEqual(report.step, 1000)
        self.assertLessEqual(sum(bytes_read), 131072)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX FIFO")
    def test_nonregular_report_is_rejected_before_opening(self):
        os.mkfifo(self.path)
        with mock.patch.object(Path, "open", side_effect=AssertionError("must not open FIFO")):
            report = _status.read_report(self.path)
        self.assertTrue(report.problem)

    def test_replaced_report_is_retried_before_using_old_measurements(self):
        self.path.write_text(HEADER + record(2000))
        replacement = self.path.with_suffix(".new")
        replacement.write_text(HEADER + record(1000))
        original_open = Path.open
        replaced = False

        class Reader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def read(reader, *args):
                nonlocal replaced
                data = reader.stream.read(*args)
                if not replaced:
                    replacement.replace(self.path)
                    replaced = True
                return data

        def open_report(path, *args, **kwargs):
            return Reader(original_open(path, *args, **kwargs))

        with mock.patch.object(Path, "open", open_report):
            report = _status.read_report(self.path)
        self.assertEqual(report.step, 1000)


class StatusDiscoveryTests(StatusFixture):
    def test_local_stage_option_can_appear_before_or_between_positionals(self):
        script = str(self.root / "mains/local.py")
        for arguments in (
            ["--stage", "step4_equilibration", "solution", "replica_2"],
            ["solution", "--stage", "step4_equilibration", "replica_2"],
            ["--stage=step4_equilibration", "solution", "replica_2"],
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(
                    _status._local_arguments(["python", script, *arguments], self.root, self.root),
                    ("solution", "replica_2", "step4_equilibration"),
                )

    def test_allocated_node_child_is_not_counted_as_a_second_local_run(self):
        proc = self.root / "proc"
        proc.mkdir()
        (proc / "uptime").write_text("1000 0\n")
        for pid, parent, script in ((40, 1, "mains/_simcalc/run_solution.py"), (41, 40, "mains/local.py")):
            directory = proc / str(pid)
            directory.mkdir()
            (directory / "exe").symlink_to(Path(sys.executable).resolve())
            (directory / "cwd").symlink_to(self.root, target_is_directory=True)
            (directory / "cmdline").write_bytes(b"python\0" + script.encode() + b"\0solution\0replica_2\0")
            fields = ["S", str(parent)] + ["0"] * 17 + ["100"] + ["0"] * 5
            (directory / "stat").write_text(f"{pid} (python) " + " ".join(fields))
        result = subprocess.CompletedProcess([], 0, f"321|solution-md-replica_2|RUNNING|{self.root}|node|00:01|01:00|N/A|None\n", "")
        with (
            mock.patch("mains._status.shutil.which", return_value="/bin/squeue"),
            mock.patch("mains._status.subprocess.run", return_value=result),
        ):
            activity = _status.read_activity(self.root, proc_root=proc, now=NOW)
        self.assertEqual([job.identifier for job in activity.executions], ["321"])

    def test_slurm_job_does_not_hide_a_separate_local_process_for_same_system(self):
        local = _status.Execution("solution", "replica_2", "PID 41", "local", "RUNNING")
        result = subprocess.CompletedProcess([], 0, f"321|solution-md-replica_2|RUNNING|{self.root}|node|00:01|01:00|N/A|None\n", "")
        with (
            mock.patch("mains._status.shutil.which", return_value="/bin/squeue"),
            mock.patch("mains._status.subprocess.run", return_value=result),
            mock.patch("mains._status._local_processes", return_value=([local], None)),
        ):
            activity = _status.read_activity(self.root)
        self.assertEqual({job.identifier for job in activity.executions}, {"321", "PID 41"})
        system = _status.collect_snapshot(self.root, activity=activity).systems[0]
        self.assertIsNone(system.eta_seconds)
        self.assertIn("Multiple active", _status.render_snapshot(_status.collect_snapshot(self.root, activity=activity)))

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "requires Linux process information")
    def test_real_local_process_is_visible_before_any_stage_outputs_exist(self):
        script = self.root / "mains/local.py"
        script.parent.mkdir()
        script.write_text("import time\nprint('ready', flush=True)\ntime.sleep(20)\n")
        child = subprocess.Popen(
            [sys.executable, "-u", str(script), "solution", "replica_2"],
            stdout=subprocess.PIPE, text=True, cwd=self.root,
        )
        try:
            self.assertEqual(child.stdout.readline(), "ready\n")
            with mock.patch("mains._status.shutil.which", return_value=None):
                activity = _status.read_activity(self.root)
            self.assertEqual([job.identifier for job in activity.executions], [f"PID {child.pid}"])
            snapshot = _status.collect_snapshot(self.root, activity=activity)
            self.assertEqual(snapshot.systems[0].state, "RUNNING")
            self.assertIn("No simulation report", _status.render_snapshot(snapshot))
        finally:
            child.terminate()
            child.communicate(timeout=5)

    def test_slurm_query_filters_user_prefix_and_project_with_one_call(self):
        result = subprocess.CompletedProcess([], 0, "\n".join([
            f"321|solution-md-replica_2|RUNNING|{self.root}|nvdfen1|1-02:00:00|5-22:00:00|2023-11-14T20:00:00|None",
            f"322|unrelated|RUNNING|{self.root}|node|00:01|01:00|N/A|None",
            "323|solution-md-other|RUNNING|/other/checkout|node|00:01|01:00|N/A|None",
        ]), "")
        with (
            mock.patch("mains._status.shutil.which", return_value="/bin/squeue"),
            mock.patch("mains._status.subprocess.run", return_value=result) as run,
        ):
            activity = _status.read_activity(self.root, proc_root=self.root / "missing-proc")
        self.assertEqual([job.identifier for job in activity.executions], ["321"])
        self.assertEqual(run.call_count, 1)
        arguments = run.call_args.args[0]
        self.assertEqual(arguments[0], "squeue")
        self.assertIn("-u", arguments)
        self.assertEqual(run.call_args.kwargs["timeout"], 5)

    def test_slurm_failure_is_explicit_and_never_submits(self):
        with (
            mock.patch("mains._status.shutil.which", return_value="/bin/squeue"),
            mock.patch("mains._status.subprocess.run", side_effect=subprocess.TimeoutExpired("squeue", 5)) as run,
        ):
            activity = _status.read_activity(self.root, proc_root=self.root / "missing-proc")
        self.assertTrue(activity.slurm_error)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][0], "squeue")

    def test_local_process_matching_ignores_wrappers_and_other_checkouts(self):
        proc = self.root / "proc"
        proc.mkdir()
        (proc / "uptime").write_text("1000 0\n")
        python = Path(sys.executable).resolve()
        for pid, executable, cwd, script in (
            (41, python, self.root, "mains/local.py"),
            (42, Path("/bin/uv"), self.root, "mains/local.py"),
            (43, python, self.root.parent, "mains/local.py"),
        ):
            directory = proc / str(pid)
            directory.mkdir()
            (directory / "exe").symlink_to(executable)
            (directory / "comm").write_text("python\n" if executable == python else "uv\n")
            (directory / "cwd").symlink_to(cwd, target_is_directory=True)
            (directory / "cmdline").write_bytes(
                b"python\0" + os.fsencode(script) + b"\0membrane\0replica with spaces\0--stage\0step6.3_equilibration\0"
            )
            fields = ["S", "1"] + ["0"] * 17 + ["100"] + ["0"] * 5
            (directory / "stat").write_text(f"{pid} (python process) " + " ".join(fields))
        with mock.patch("mains._status.shutil.which", return_value=None):
            activity = _status.read_activity(self.root, proc_root=proc, now=NOW)
        self.assertEqual([job.identifier for job in activity.executions], ["PID 41"])
        self.assertEqual(activity.executions[0].system_id, "replica with spaces")
        self.assertEqual(activity.executions[0].stage, "step6.3_equilibration")
        self.assertLess(activity.executions[0].started_at, NOW)

    def test_unreadable_unrelated_executable_does_not_make_local_activity_unknown(self):
        proc = self.root / "proc"
        proc.mkdir()
        (proc / "uptime").write_text("1000 0\n")
        directory = proc / "41"
        directory.mkdir()
        (directory / "comm").write_text("chrome-sandbox\n")
        (directory / "cmdline").write_bytes(b"/bin/chrome-sandbox\0")
        original = Path.resolve

        def resolve(path, *args, **kwargs):
            if path == directory / "exe":
                raise PermissionError("protected process")
            return original(path, *args, **kwargs)

        with (
            mock.patch("mains._status.shutil.which", return_value=None),
            mock.patch.object(Path, "resolve", resolve),
        ):
            activity = _status.read_activity(self.root, proc_root=proc, now=NOW)
        self.assertIsNone(activity.local_error)


class StatusFormattingTests(StatusFixture):
    def test_comparison_rows_precede_details_and_completed_has_no_detail_block(self):
        self.stage("step4_equilibration", complete=True)
        stage = self.stage()
        self.csv(stage, record(160_000_000, "13.4"))
        self.stage("step4_equilibration", system="done", complete=True)
        self.stage(system="done", complete=True)
        output = _status.render_snapshot(self.snapshot(self.job()), width=160)
        self.assertIn("Job/PID", output)
        self.assertIn("Stage %", output)
        self.assertIn("Report age", output)
        self.assertIn("320 / 500 ns", output)
        self.assertIn("160,000,000 / 250,000,000 steps", output)
        self.assertLess(output.index("solution/done"), output.index("DETAILS"))
        self.assertEqual(output.count("solution/done"), 1)

    def test_long_names_are_preserved_in_both_table_and_narrow_output(self):
        name = "abeta42_7q4m_8bfa_replica_2_very_long_system_name"
        self.stage("step4_equilibration", system=name)
        snapshot = self.snapshot()
        wide = _status.render_snapshot(snapshot, width=200)
        narrow = _status.render_snapshot(snapshot, width=55)
        self.assertIn(name, wide)
        self.assertIn(name, narrow)
        self.assertIn("Stage %", narrow)
        self.assertNotIn("Job/PID  State", narrow)

    def test_plain_output_has_no_escape_codes_and_colour_does_not_change_alignment(self):
        self.stage("step4_equilibration")
        snapshot = self.snapshot(self.job())
        plain = _status.render_snapshot(snapshot, width=160)
        coloured = _status.render_snapshot(snapshot, width=160, color=True)
        self.assertNotIn("\x1b", plain)
        self.assertIn("\x1b[", coloured)
        import re
        self.assertEqual(re.sub(r"\x1b\[[0-9;]*m", "", coloured), plain)


class StatusCliTests(unittest.TestCase):
    def test_keyboard_interrupt_leaves_simulations_running(self):
        from mains import status
        with (
            mock.patch.object(status, "collect_snapshot", side_effect=KeyboardInterrupt),
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            result = status.main([])
        self.assertEqual(result, 130)
        self.assertIn("simulations continue", errors.getvalue())

    def test_no_site_packages_or_side_effects_on_an_entire_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(PROJECT / "mains", root / "mains", ignore=shutil.ignore_patterns("__pycache__", "logs"))
            executable = Path(sys.executable).resolve()
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            result = subprocess.run(
                [str(executable), "-S", str(root / "mains/status.py")], cwd="/tmp",
                env={**os.environ, "PATH": "", "NO_COLOR": "1"},
                capture_output=True, text=True, check=False,
            )
            after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No simulations", result.stdout)
        self.assertEqual(before, after)
        self.assertNotIn("\x1b", result.stdout + result.stderr)

    def test_cli_and_helpers_parse_as_python_38(self):
        for name in ("status.py", "_status.py", "_console.py"):
            with self.subTest(name=name):
                ast.parse((PROJECT / "mains" / name).read_text(), feature_version=(3, 8))


if __name__ == "__main__":
    unittest.main()
