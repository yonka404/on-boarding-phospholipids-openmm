"""Exercise the launchers with filesystem fixtures and fake Slurm commands."""

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


class SimcalcLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project with spaces"
        self.root.mkdir()
        (self.root / "simcalc").mkdir()
        for name in ("main.sh", "run_solution.slurm", "common.sh"):
            source = PROJECT / "simcalc" / name
            if source.exists():
                shutil.copyfile(source, self.root / "simcalc" / name)
        (self.root / "simcalc/validate_solution.py").touch()
        (self.root / "mains").mkdir()
        (self.root / "mains/run_solution_sweep.py").touch()
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
import json, os, pathlib, sys, time
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
    if mode == "duplicate": print("203|solution-md-test_replica_2|RUNNING")
    elif mode == "other_job": print("203|solution-md-other_system|RUNNING")
elif name == "sbatch":
    if "--test-only" in args:
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
    else: print("321")
elif name == "mkdir":
    if os.environ.get("FAKE_APPEAR_ON_LOCK") and args[-1].endswith(".test_replica_2.lock"):
        pathlib.Path(os.environ["FAKE_OUTPUT"]).mkdir()
    os.execv("/usr/bin/mkdir", ["mkdir", *args])
elif name == "getconf": print("glibc " + os.environ.get("FAKE_GLIBC", "2.39"))
elif name == "python":
    if "validate_solution.py" in " ".join(args) and mode == "allocation_fail":
        print("GPU allocation invalid", file=sys.stderr)
        sys.exit(1)
    print("GPU invocation " + os.environ.get("OPENMM_PLATFORM", "") + " " + os.environ.get("OPENMM_DEVICE_INDEX", ""))
elif name == "srun":
    os.execv(args[0], args)
""")
        tool.chmod(0o755)
        for name in (
            "scontrol",
            "sinfo",
            "squeue",
            "sbatch",
            "module",
            "getconf",
            "srun",
            "mkdir",
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

    def run_script(self, script="main.sh", args=None, mode=""):
        return subprocess.run(
            [
                "bash",
                str(self.root / "simcalc" / script),
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

    def test_queues_on_busy_nodes_and_forwards_exact_system(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("321", result.stdout)
        submissions = [a for a in self.invocations("sbatch") if "--parsable" in a]
        self.assertEqual(len(submissions), 1)
        self.assertEqual(submissions[0][-1], self.system_id)
        self.assertFalse(any("nodelist" in arg for arg in submissions[0]))
        self.assertFalse((self.outputs / self.system_id).exists())

    def test_check_only_never_submits(self):
        result = self.run_script(args=["--check-only", self.system_id])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.invocations("sbatch"))
        self.assertFalse(any("--parsable" in a for a in self.invocations("sbatch")))

    def test_missing_estimate_still_submits(self):
        result = self.run_script(mode="no_estimate")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum("--parsable" in a for a in self.invocations("sbatch")), 1)

    def test_bad_arguments_do_not_contact_slurm(self):
        for args in ([], ["../bad"], ["."], [".."], ["a", "b"]):
            with self.subTest(args=args):
                self.assertNotEqual(self.run_script(args=args).returncode, 0)
        self.assertEqual(self.invocations("sbatch"), [])

    def test_missing_input_and_python_fail_before_submission(self):
        for path in (
            self.inputs / "step5_production.inp",
            self.root / ".venv/bin/python",
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
            self.root / "simcalc/logs" / f".{self.system_id}.submit.lock",
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
        ):
            with self.subTest(mode=mode):
                result = self.run_script(mode=mode)
                self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(any("--parsable" in a for a in self.invocations("sbatch")))

    def test_submit_failure_releases_own_submission_lock(self):
        self.assertNotEqual(self.run_script(mode="submit_fail").returncode, 0)
        self.assertFalse(
            (self.root / "simcalc/logs" / f".{self.system_id}.submit.lock").exists()
        )
        self.assertEqual(self.run_script().returncode, 0)

    def test_unwritable_log_directory_prevents_submission(self):
        logs = self.root / "simcalc/logs"
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
                ["-u", "simcalc/validate_solution.py", self.system_id],
                ["-u", "mains/run_solution_sweep.py", self.system_id],
            ],
        )
        self.assertIn("GPU invocation CUDA 0", result.stdout)
        self.assertFalse((self.outputs / f".{self.system_id}.lock").exists())

    def test_allocation_validation_failure_prevents_simulation_and_cleans_lock(self):
        result = self.run_script(script="run_solution.slurm", mode="allocation_fail")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.invocations("python")), 1)
        self.assertFalse((self.outputs / self.system_id).exists())
        self.assertFalse((self.outputs / f".{self.system_id}.lock").exists())

    def test_old_glibc_fails_before_python(self):
        self.env["FAKE_GLIBC"] = "2.31"
        self.assertNotEqual(self.run_script(script="run_solution.slurm").returncode, 0)
        self.assertEqual(self.invocations("python"), [])

    def test_job_cannot_run_outside_allocation(self):
        self.env.pop("SLURM_JOB_ID")
        self.assertNotEqual(self.run_script(script="run_solution.slurm").returncode, 0)
        self.assertEqual(self.invocations("python"), [])

    def test_output_appearing_during_lock_acquisition_is_protected(self):
        self.env["FAKE_APPEAR_ON_LOCK"] = "1"
        self.env["FAKE_OUTPUT"] = str(self.outputs / self.system_id)
        result = self.run_script(script="run_solution.slurm")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.invocations("python"), [])
        self.assertTrue((self.outputs / self.system_id).exists())
        self.assertFalse((self.outputs / f".{self.system_id}.lock").exists())

    def test_simultaneous_submission_is_blocked(self):
        marker = Path(self.temp.name) / "predicting"
        env = dict(self.env, FAKE_MODE="slow_prediction", FAKE_MARKER=str(marker))
        with subprocess.Popen(
            ["bash", str(self.root / "simcalc/main.sh"), self.system_id],
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
