"""Exercise the launchers with filesystem fixtures and fake Slurm commands."""

import ast
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).resolve().parents[2]


class SimcalcLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project with spaces"
        self.root.mkdir()
        (self.root / "mains/_simcalc").mkdir(parents=True)
        for name in (
            "__init__.py", "_common.py", "local.py", "simcalc.py",
            "_simcalc/__init__.py", "_simcalc/submit.py", "_simcalc/common.py",
            "_simcalc/run_solution.py", "_simcalc/run_solution.slurm",
        ):
            source = PROJECT / "mains" / name
            if source.exists():
                shutil.copyfile(source, self.root / "mains" / name)
        (self.root / "mains/_simcalc/validate_solution.py").touch()
        for name in ("pyproject.toml", "uv.lock"):
            (self.root / name).touch()
        self.system_id = "test_replica_2"
        self.inputs = (
            self.root / "data/inputs/openmm_native/solution" / self.system_id / "openmm"
        )
        self.inputs.mkdir(parents=True)
        for name in (
            "step3_input.psf",
            "step3_input.pdb",
            "step3_input.crd",
            "step4_equilibration.inp",
            "step5_production.inp",
            "sysinfo.dat",
            "toppar.str",
        ):
            (self.inputs / name).write_text("test fixture\n")
        self.outputs = self.root / "data/outputs/openmm_native/solution"
        self.outputs.mkdir(parents=True)
        self.bin = Path(self.temp.name) / "bin"
        self.bin.mkdir()
        self.calls = Path(self.temp.name) / "calls.jsonl"
        tool = self.bin / "tool"
        tool.write_text("""#!/usr/bin/python3
import json, os, pathlib, shutil, signal, sys, time
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["FAKE_CALLS"], "a") as f:
    f.write(json.dumps([name, args]) + "\\n")
mode = os.environ.get("FAKE_MODE", "")
if name == "scontrol":
    if mode == "controller_down":
        print("Slurmctld(primary) is DOWN")
        sys.exit(1)
    if args[0] == "ping": print("Slurmctld(primary) is UP")
    else:
        state = "DOWN" if mode == "partition_down" else "UP"
        limit = "1-00:00:00" if mode == "short_walltime" else "8-00:00:00"
        print(f"PartitionName=gpu State={state} MaxTime={limit}")
elif name == "sinfo":
    gpu = "(null)" if mode == "no_gpu" else "gpu:a4000:4"
    cpus = "2" if mode == "few_cpus" else "32"
    mem = "8000" if mode == "little_ram" else "64000"
    state = "down" if mode == "nodes_down" else "allocated"
    print(f"nvdfen1|{state}|{gpu}|{cpus}|{mem}")
    print(f"nvdfen2|{state}|{gpu}|{cpus}|{mem}")
elif name == "squeue":
    if mode == "queue_fail": sys.exit(1)
    if mode == "bad_queue": print("not a valid queue row")
    if mode == "duplicate": print("203|solution-md-test_replica_2|RUNNING")
    elif mode == "other_job": print("203|solution-md-other_system|RUNNING")
elif name == "sbatch":
    if "--test-only" in args:
        if mode == "new_output":
            pathlib.Path(os.environ["FAKE_OUTPUT"]).mkdir()
        if mode == "removed_output":
            shutil.rmtree(os.environ["FAKE_OUTPUT"])
        if mode == "removed_stage":
            shutil.rmtree(os.environ["FAKE_STAGE"])
        if mode == "slow_prediction":
            pathlib.Path(os.environ["FAKE_MARKER"]).touch()
            time.sleep(0.5)
        if mode == "no_estimate":
            print("Start estimate unavailable", file=sys.stderr)
            sys.exit(1)
        print("Job would start later")
    elif mode == "submit_fail":
        print("Submission refused", file=sys.stderr)
        sys.exit(1)
    elif mode == "bad_reply": print("unrecognized reply")
    elif mode == "federated": print("321;simcalc")
    else: print("321")
elif name == "module":
    if mode == "module_fail": sys.exit(1)
elif name == "getconf": print("glibc " + os.environ.get("FAKE_GLIBC", "2.39"))
elif name == "python":
    if "validate_solution.py" in " ".join(args) and mode == "allocation_fail":
        print("GPU allocation invalid", file=sys.stderr)
        sys.exit(1)
    if "local.py" in " ".join(args):
        if mode == "sweep_fail": sys.exit(17)
        if mode == "long_sweep":
            def interrupted(signum, frame):
                lock = pathlib.Path(os.environ["FAKE_LOCK"])
                pathlib.Path(os.environ["FAKE_STOPPED"]).write_text(str(lock.exists()))
                sys.exit(128 + signum)
            signal.signal(signal.SIGTERM, interrupted)
            signal.signal(signal.SIGINT, interrupted)
            pathlib.Path(os.environ["FAKE_MARKER"]).touch()
            time.sleep(15)
    print("GPU invocation " + os.environ.get("OPENMM_PLATFORM", "") + " " + os.environ.get("OPENMM_DEVICE_INDEX", "") + " mask=" + os.environ.get("CUDA_VISIBLE_DEVICES", ""))
""")
        tool.chmod(0o755)
        for name in (
            "scontrol",
            "sinfo",
            "squeue",
            "sbatch",
            "module",
            "getconf",
        ):
            (self.bin / name).symlink_to(tool)
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(tool)
        self.env = dict(
            os.environ,
            PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
            FAKE_CALLS=str(self.calls),
            SLURM_SUBMIT_DIR=str(self.root),
            SLURM_JOB_ID="203",
            CUDA_VISIBLE_DEVICES="2",
        )

    def run_script(self, script="simcalc.py", args=None, mode=""):
        return subprocess.run(
            [
                *(["bash"] if script.endswith(".slurm") else [sys.executable, "-S"]),
                str(self.root / "mains" / (script if script == "simcalc.py" else "_simcalc/" + script)),
                *(args if args is not None else [self.system_id]),
            ],
            env=dict(self.env, FAKE_MODE=mode),
            cwd=self.temp.name,
            capture_output=True,
            text=True,
            check=False,
        )

    def invocations(self, name):
        if not self.calls.exists():
            return []
        return [
            args
            for command, args in map(json.loads, self.calls.read_text().splitlines())
            if command == name
        ]

    def completed_stage(self, name="step4_equilibration"):
        stage = self.outputs / self.system_id / name
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "final_state.xml").write_text("<state />")
        (stage / "final_coordinates.pdb").write_text("ATOM")
        return stage

    def interrupted_stage(self, name="step4_equilibration"):
        stage = self.outputs / self.system_id / name
        stage.mkdir(parents=True, exist_ok=True)
        for filename in ("checkpoint.chk", "state_data.csv", "trajectory.dcd"):
            (stage / filename).write_bytes(b"validated on allocated GPU")
        return stage

    def test_checkpoint_recovery_is_submitted_and_reaches_cuda_workflow(self):
        stage = self.interrupted_stage()
        before = {path.name: path.read_bytes() for path in stage.iterdir()}
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        submission = next(a for a in self.invocations("sbatch") if "--parsable" in a)
        self.assertEqual(submission[-3:], ["--mode", "resume", self.system_id])
        result = self.run_script(
            script="run_solution.slurm", args=["--mode", "resume", self.system_id]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GPU invocation CUDA 0 mask=2", result.stdout)
        self.assertEqual(before, {path.name: path.read_bytes() for path in stage.iterdir()})
        self.assertFalse((self.outputs / f".{self.system_id}.lock").exists())

    def test_continuation_between_stages_is_submitted_as_resume(self):
        self.completed_stage()
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("step5_production", result.stderr)
        submission = next(a for a in self.invocations("sbatch") if "--parsable" in a)
        self.assertEqual(submission[-3:], ["--mode", "resume", self.system_id])

    def test_completed_run_is_a_noop_without_slurm(self):
        self.completed_stage()
        self.completed_stage("step5_production")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("complete", result.stderr.lower())
        self.assertFalse(self.calls.exists())

    def test_checkpoint_takes_precedence_during_finalization(self):
        self.completed_stage()
        self.interrupted_stage()
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("step4_equilibration", result.stderr)

    def test_missing_recovery_files_and_invalid_stage_order_are_preserved(self):
        output = self.outputs / self.system_id
        cases = [
            {"step4_equilibration": ("checkpoint.chk", "state_data.csv")},
            {"step4_equilibration": ("state_data.csv", "trajectory.dcd")},
            {"step5_production": ("final_state.xml", "final_coordinates.pdb")},
            {"unexpected": ("something",)},
            {"step4_equilibration": ()},
            {},
            {
                "step4_equilibration": ("checkpoint.chk", "state_data.csv", "trajectory.dcd"),
                "step5_production": ("final_state.xml", "final_coordinates.pdb"),
            },
        ]
        for stages in cases:
            with self.subTest(stages=stages):
                output.mkdir(exist_ok=True)
                for name, files in stages.items():
                    stage = output / name
                    stage.mkdir()
                    for filename in files:
                        (stage / filename).write_text("preserve me")
                before = sorted(str(path.relative_to(output)) for path in output.rglob("*"))
                result = self.run_script()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("inspect", result.stderr.lower())
                self.assertEqual(before, sorted(str(path.relative_to(output)) for path in output.rglob("*")))
                shutil.rmtree(output)
        self.assertEqual(self.invocations("sbatch"), [])

    def test_queued_resume_refuses_missing_output(self):
        result = self.run_script(
            script="run_solution.slurm", args=["--mode", "resume", self.system_id]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing", result.stderr.lower())
        self.assertEqual(self.invocations("python"), [])

    def test_queued_fresh_refuses_even_recoverable_output(self):
        self.interrupted_stage()
        result = self.run_script(script="run_solution.slurm")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)
        self.assertEqual(self.invocations("python"), [])

    def test_completed_queued_recovery_does_not_start_python(self):
        self.completed_stage()
        self.completed_stage("step5_production")
        result = self.run_script(
            script="run_solution.slurm", args=["--mode", "resume", self.system_id]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.invocations("python"), [])

    def test_selected_checkpoint_stage_disappearing_while_queued_is_protected(self):
        self.completed_stage()
        interrupted = self.interrupted_stage("step5_production")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        submission = next(a for a in self.invocations('sbatch') if '--parsable' in a)
        batch_index = submission.index(str(self.root / 'mains/_simcalc/run_solution.slurm'))
        shutil.rmtree(interrupted)
        result = self.run_script(script='run_solution.slurm', args=submission[batch_index + 1:])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('step5_production', result.stderr)
        self.assertIn('missing', result.stderr.lower())
        self.assertEqual(self.invocations('python'), [])
        self.assertFalse(interrupted.exists())

    def test_selected_checkpoint_stage_disappearing_before_submission_is_protected(self):
        self.completed_stage()
        interrupted = self.interrupted_stage('step5_production')
        self.env['FAKE_STAGE'] = str(interrupted)
        result = self.run_script(mode='removed_stage')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing', result.stderr.lower())
        self.assertFalse(any('--parsable' in a for a in self.invocations('sbatch')))

    def test_selected_checkpoint_stage_can_finish_while_queued(self):
        self.completed_stage()
        interrupted = self.interrupted_stage('step5_production')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        submission = next(a for a in self.invocations('sbatch') if '--parsable' in a)
        batch_index = submission.index(str(self.root / 'mains/_simcalc/run_solution.slurm'))
        (interrupted / 'checkpoint.chk').unlink()
        self.completed_stage('step5_production')
        result = self.run_script(script='run_solution.slurm', args=submission[batch_index + 1:])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.invocations('python'), [])

    def test_queues_on_busy_nodes_and_forwards_exact_system(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("321", result.stdout)
        self.assertIn(str(self.root / "mains/logs/test_replica_2-321.log"), result.stdout)
        submissions = [a for a in self.invocations("sbatch") if "--parsable" in a]
        self.assertEqual(len(submissions), 1)
        self.assertEqual(submissions[0][-1], self.system_id)
        self.assertEqual(submissions[0][-3:-1], ["--mode", "fresh"])
        self.assertFalse(any("nodelist" in arg for arg in submissions[0]))
        self.assertFalse((self.outputs / self.system_id).exists())
        self.assertEqual(self.invocations("python"), [])
        self.assertTrue(self.invocations("scontrol"))
        self.assertTrue(self.invocations("squeue"))
        self.assertTrue(self.invocations("sinfo"))
        self.assertTrue(any("--test-only" in a for a in self.invocations("sbatch")))

    def test_removed_check_only_flag_is_rejected_without_contacting_slurm(self):
        result = self.run_script(args=["--check-only", self.system_id])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrecognized arguments", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_missing_estimate_still_submits(self):
        result = self.run_script(mode="no_estimate")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum("--parsable" in a for a in self.invocations("sbatch")), 1)

    def test_bad_arguments_do_not_contact_slurm(self):
        for args in ([], ["../bad"], ["."], [".."], ["a", "b"], ["a b"], ["a;touch x"]):
            with self.subTest(args=args):
                self.assertNotEqual(self.run_script(args=args).returncode, 0)
        self.assertEqual(self.invocations("sbatch"), [])

    def test_missing_input_and_python_fail_before_submission(self):
        for path in (
            self.inputs / "step5_production.inp",
            self.root / ".venv/bin/python",
            self.root / "mains/_simcalc/run_solution.py",
        ):
            with self.subTest(path=path):
                saved = path.read_bytes() if not path.is_symlink() else None
                target = path.resolve() if path.is_symlink() else None
                path.unlink()
                self.assertNotEqual(self.run_script().returncode, 0)
                if target:
                    path.symlink_to(target)
                else:
                    path.write_bytes(saved)
        self.assertEqual(self.invocations("sbatch"), [])

    def test_existing_output_including_dangling_link_is_protected(self):
        output = self.outputs / self.system_id
        output.symlink_to(self.root / "nonexistent")
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertTrue(output.is_symlink())
        self.assertEqual(self.invocations("sbatch"), [])

    def test_execution_and_submission_locks_are_not_removed(self):
        for lock in (
            self.outputs / f".{self.system_id}.lock",
            self.root / "mains/logs" / f".{self.system_id}.submit.lock",
        ):
            lock.mkdir(parents=True)
            self.assertNotEqual(self.run_script().returncode, 0)
            self.assertTrue(lock.exists())
            lock.rmdir()
        self.assertFalse(any("--parsable" in a for a in self.invocations("sbatch")))

    def test_duplicate_job_refused_but_other_system_is_allowed(self):
        self.assertNotEqual(self.run_script(mode="duplicate").returncode, 0)
        self.assertEqual(self.run_script(mode="other_job").returncode, 0)
        self.assertEqual(sum("--parsable" in a for a in self.invocations("sbatch")), 1)

    def test_slurm_connection_and_capacity_errors_prevent_submission(self):
        for mode in (
            "controller_down",
            "partition_down",
            "no_gpu",
            "few_cpus",
            "little_ram",
            "short_walltime",
            "nodes_down",
            "queue_fail",
        ):
            with self.subTest(mode=mode):
                result = self.run_script(mode=mode)
                self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(any("--parsable" in a for a in self.invocations("sbatch")))

    def test_submit_failure_releases_own_submission_lock(self):
        self.assertNotEqual(self.run_script(mode="submit_fail").returncode, 0)
        self.assertFalse(
            (self.root / "mains/logs" / f".{self.system_id}.submit.lock").exists()
        )
        self.assertEqual(self.run_script().returncode, 0)

    def test_unwritable_log_directory_prevents_submission(self):
        logs = self.root / "mains/logs"
        logs.mkdir()
        logs.chmod(0o555)
        self.addCleanup(logs.chmod, 0o755)
        result = self.run_script()
        if os.geteuid() == 0:
            self.skipTest("root can write to read-only permission fixtures")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any("--parsable" in a for a in self.invocations("sbatch")))

    def test_gpu_job_runs_only_requested_system_and_forces_cuda(self):
        result = self.run_script(script="run_solution.slurm")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.invocations("python")
        self.assertEqual(
            calls,
            [
                ["-u", "mains/_simcalc/validate_solution.py", self.system_id],
                ["-u", "mains/local.py", "solution", self.system_id],
            ],
        )
        self.assertIn("GPU invocation CUDA 0", result.stdout)
        self.assertIn("mask=2", result.stdout)
        self.assertFalse((self.outputs / f".{self.system_id}.lock").exists())

    def test_allocation_validation_failure_prevents_simulation_and_cleans_lock(self):
        result = self.run_script(script="run_solution.slurm", mode="allocation_fail")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.invocations("python")), 1)
        self.assertFalse((self.outputs / self.system_id).exists())
        self.assertFalse((self.outputs / f".{self.system_id}.lock").exists())

    def test_old_glibc_fails_before_project_python(self):
        self.env["FAKE_GLIBC"] = "2.31"
        self.assertNotEqual(self.run_script(script="run_solution.slurm").returncode, 0)
        self.assertEqual(self.invocations("python"), [])

    def test_job_cannot_run_outside_allocation(self):
        self.env.pop("SLURM_JOB_ID")
        self.assertNotEqual(self.run_script(script="run_solution.slurm").returncode, 0)
        self.assertEqual(self.invocations("python"), [])

    def test_output_appearing_during_lock_acquisition_is_protected(self):
        from mains._simcalc import common, run_solution

        original_mkdir = Path.mkdir
        output = self.outputs / self.system_id

        def mkdir(path, *args, **kwargs):
            if path.name == f".{self.system_id}.lock":
                original_mkdir(output)
            return original_mkdir(path, *args, **kwargs)

        original_directory = Path.cwd()
        self.addCleanup(os.chdir, original_directory)
        with (
            mock.patch.dict(os.environ, self.env),
            mock.patch.object(Path, "mkdir", mkdir),
            self.assertRaisesRegex(common.LauncherError, "Output already exists"),
        ):
            run_solution.run_job(self.system_id)
        self.assertEqual(self.invocations("python"), [])
        self.assertTrue((self.outputs / self.system_id).exists())
        self.assertFalse((self.outputs / f".{self.system_id}.lock").exists())

    def test_simultaneous_submission_is_blocked(self):
        self.assert_simultaneous_submission_is_blocked()

    def test_simultaneous_recovery_submission_is_blocked(self):
        self.interrupted_stage()
        self.assert_simultaneous_submission_is_blocked()

    def assert_simultaneous_submission_is_blocked(self):
        marker = Path(self.temp.name) / "predicting"
        env = dict(self.env, FAKE_MODE="slow_prediction", FAKE_MARKER=str(marker))
        with subprocess.Popen(
            [sys.executable, "-S", str(self.root / "mains/simcalc.py"), self.system_id],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as first:
            deadline = time.monotonic() + 3
            while (
                not marker.exists()
                and time.monotonic() < deadline
                and first.poll() is None
            ):
                time.sleep(0.01)
            self.assertTrue(marker.exists(), "first launcher did not reach prediction")
            second = self.run_script()
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Submission lock", second.stderr)
            stdout, stderr = first.communicate(timeout=3)
            self.assertEqual(first.returncode, 0, stdout + stderr)
        self.assertEqual(sum("--parsable" in a for a in self.invocations("sbatch")), 1)

    def test_output_mode_changes_during_prediction_prevent_submission(self):
        self.env['FAKE_OUTPUT'] = str(self.outputs / self.system_id)
        result = self.run_script(mode='new_output')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Output already exists', result.stderr)
        self.interrupted_stage()
        result = self.run_script(mode='removed_output')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing', result.stderr)
        self.assertFalse(any('--parsable' in a for a in self.invocations('sbatch')))
        self.assertFalse((self.root / 'mains/logs' / f'.{self.system_id}.submit.lock').exists())

    def test_recovery_results_removed_during_execution_lock_acquisition_stop_job(self):
        from mains._simcalc import common, run_solution

        self.completed_stage()
        interrupted = self.interrupted_stage('step5_production')
        mkdir = Path.mkdir

        def acquire(path, *args, **kwargs):
            if path.name == f'.{self.system_id}.lock':
                shutil.rmtree(interrupted)
            return mkdir(path, *args, **kwargs)

        original_directory = Path.cwd()
        self.addCleanup(os.chdir, original_directory)
        with (
            mock.patch.dict(os.environ, self.env),
            mock.patch.object(Path, 'mkdir', acquire),
            self.assertRaisesRegex(common.LauncherError, 'missing'),
        ):
            run_solution.run_job(self.system_id, 'resume', 'step5_production')
        self.assertEqual(self.invocations('python'), [])
        self.assertFalse((self.outputs / f'.{self.system_id}.lock').exists())

    def test_recovery_duplicate_job_is_refused(self):
        self.interrupted_stage()
        result = self.run_script(mode='duplicate')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already RUNNING', result.stderr)
        self.assertEqual(self.invocations('sbatch'), [])

    def test_missing_slurm_command_prevents_submission(self):
        (self.bin / "sinfo").unlink()
        self.env["PATH"] = str(self.bin)
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sinfo", result.stderr)
        self.assertEqual(self.invocations("sbatch"), [])

    def test_malformed_queue_fails_clearly_without_submitting(self):
        result = self.run_script(mode="bad_queue")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("queue", result.stderr.lower())
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(self.invocations("sbatch"), [])

    def test_checks_use_resource_requests_from_batch_file(self):
        batch = self.root / "mains/_simcalc/run_solution.slurm"
        original = batch.read_text()
        for old, new in (
            ("--mem=16G", "--mem=80G"),
            ("--cpus-per-task=4", "--cpus-per-task=64"),
            ("--time=7-00:00:00", "--time=9-00:00:00"),
        ):
            with self.subTest(request=new):
                batch.write_text(original.replace(old, new))
                result = self.run_script()
                self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.invocations("sbatch"), [])

    def test_node_rechecks_inputs_and_existing_output_before_project_python(self):
        (self.inputs / "step3_input.psf").write_text("")
        result = self.run_script(script="run_solution.slurm")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("step3_input.psf", result.stderr)
        (self.inputs / "step3_input.psf").write_text("restored fixture")
        (self.outputs / self.system_id).mkdir()
        result = self.run_script(script="run_solution.slurm")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)
        self.assertEqual(self.invocations("python"), [])

    def test_bootstrap_syntax_supports_python_38(self):
        for name in (
            "__init__.py", "_common.py", "simcalc.py", "_simcalc/__init__.py",
            "_simcalc/submit.py", "_simcalc/common.py", "_simcalc/run_solution.py",
        ):
            with self.subTest(name=name):
                ast.parse(
                    (PROJECT / "mains" / name).read_text(), feature_version=(3, 8)
                )

    def test_help_works_with_system_python_without_site_packages(self):
        for name in ("simcalc.py", "_simcalc/run_solution.py"):
            result = subprocess.run(
                ["/usr/bin/python3", "-S", str(self.root / "mains" / name), "--help"],
                cwd=self.temp.name,
                env=self.env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("system_id", result.stdout)
        self.assertFalse(self.calls.exists())

    def test_malformed_reply_warns_without_resubmitting(self):
        result = self.run_script(mode="bad_reply")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("squeue", result.stderr)
        self.assertEqual(sum("--parsable" in a for a in self.invocations("sbatch")), 1)

    def test_federated_reply_extracts_job_id(self):
        result = self.run_script(mode="federated")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("squeue -j 321", result.stdout)
        self.assertIn("test_replica_2-321.log", result.stdout)

    def test_module_failure_prevents_project_python(self):
        result = self.run_script(script="run_solution.slurm", mode="module_fail")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.invocations("python"), [])

    def test_sweep_failure_keeps_exit_status_and_releases_lock(self):
        result = self.run_script(script="run_solution.slurm", mode="sweep_fail")
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertFalse((self.outputs / f".{self.system_id}.lock").exists())

    def test_interruption_stops_sweep_before_releasing_execution_lock(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signum=signum):
                marker = Path(self.temp.name) / f"running-{signum}"
                stopped = Path(self.temp.name) / f"stopped-{signum}"
                lock = self.outputs / f".{self.system_id}.lock"
                env = dict(
                    self.env,
                    FAKE_MODE="long_sweep",
                    FAKE_MARKER=str(marker),
                    FAKE_STOPPED=str(stopped),
                    FAKE_LOCK=str(lock),
                )
                with subprocess.Popen(
                    [
                        "bash",
                        str(self.root / "mains/_simcalc/run_solution.slurm"),
                        self.system_id,
                    ],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ) as job:
                    try:
                        deadline = time.monotonic() + 5
                        while (
                            not marker.exists()
                            and time.monotonic() < deadline
                            and job.poll() is None
                        ):
                            time.sleep(0.01)
                        self.assertTrue(marker.exists(), "sweep did not start")
                        self.assertTrue(lock.exists())
                        job.send_signal(signum)
                        stdout, stderr = job.communicate(timeout=5)
                        self.assertEqual(job.returncode, 128 + signum, stdout + stderr)
                        self.assertEqual(stopped.read_text(), "True")
                        self.assertFalse(lock.exists())
                    finally:
                        if job.poll() is None:
                            job.kill()
                            job.communicate(timeout=20)

    def test_interruption_during_spawn_stops_child_before_releasing_lock(self):
        from mains._simcalc import common

        original_popen = subprocess.Popen
        children = []

        def spawn_then_interrupt(*args, **kwargs):
            child = original_popen(*args, **kwargs)
            children.append(child)
            os.kill(os.getpid(), signal.SIGTERM)
            return child

        lock = self.outputs / f".{self.system_id}.lock"
        try:
            with (
                self.assertRaises(common.Interrupted),
                common.interruption_handlers(),
                common.directory_lock(lock, "Execution"),
                mock.patch("mains._simcalc.common.subprocess.Popen", spawn_then_interrupt),
            ):
                try:
                    common.run_command(
                        [sys.executable, "-c", "import time; time.sleep(15)"],
                        "Child failed",
                    )
                except common.Interrupted:
                    self.assertTrue(lock.exists())
                    self.assertIsNotNone(
                        children[0].poll(), "child must stop before lock cleanup"
                    )
                    raise
            self.assertFalse(lock.exists())
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
