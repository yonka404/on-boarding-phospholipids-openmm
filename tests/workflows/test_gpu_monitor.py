"""GPU updates across the simulation process and the read-only status command."""

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from charmm_gui_md import solution
from mains import _status, local


class JobGpuTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.logs = self.root / "mains/logs"
        self.logs.mkdir(parents=True)
        self.sample = self.logs / "replica-123.gpu.json"

    def snapshot(self, started, *, job_id="123", state="RUNNING", now=None):
        job = _status.Execution("solution", "replica", job_id, "slurm", state, started_at=started)
        return _status.collect_snapshot(
            self.root, now=now, activity=_status.Activity((job,), slurm_present=True),
        ).systems[0]

    def test_allocated_launcher_publishes_for_its_job_and_warns_once(self):
        from mains import _gpu_monitor
        from mains._gpu import GpuUsage

        started = time.time() - 10
        second_read = threading.Event()
        pids = []

        def measure(pid):
            pids.append(pid)
            if len(pids) == 2:
                second_read.set()
            return GpuUsage(2, 3 * 1024**3)

        def running(*, inputs_dir, outputs_dir):
            self.assertTrue(second_read.wait(timeout=3), "GPU polling did not run")
            system = self.snapshot(started)
            self.assertEqual((system.gpu.count, system.gpu.memory_bytes), (2, 3 * 1024**3))
            output = _status.render_snapshot(_status.Snapshot(time.time(), (system,)), width=55)
            self.assertIn("2!", output)
            self.assertIn("3 GiB", output)
            self.assertIn("more than one GPU", output)
            self.assertIsNone(self.snapshot(started, job_id="124").gpu.count)
            self.assertEqual(self.snapshot(started, state="PENDING").gpu.count, 0)

        with (
            mock.patch.object(local, "PROJECT_ROOT", self.root),
            mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_JOB_NAME": "solution-md-replica"}),
            mock.patch.object(_gpu_monitor, "measure_gpu", side_effect=measure),
            mock.patch.object(_gpu_monitor, "SAMPLE_INTERVAL", 0.01),
            mock.patch.object(solution, "run_protocol_sweep", side_effect=running),
            self.assertLogs("mains._gpu_monitor", level="WARNING") as logs,
        ):
            local.main(["solution", "replica"])
        self.assertEqual(set(pids), {os.getpid()})
        self.assertEqual(len(logs.output), 1)
        self.assertIn("2 GPUs", logs.output[0])
        self.assertEqual(list(self.logs.iterdir()), [])

    def test_stale_restarted_or_damaged_samples_never_supply_current_usage(self):
        now = 1_700_000_000
        data = {"run_started_at": now - 120, "sampled_at": now - 1, "count": 1,
                "memory_bytes": 512 * 1024**2, "problem": None}
        self.sample.write_text(json.dumps(data))
        self.assertEqual(self.snapshot(now - 150, now=now).gpu.memory_bytes, 512 * 1024**2)
        self.assertIsNone(self.snapshot(now - 150, state="REQUEUED", now=now).gpu.count)
        cases = [
            (json.dumps(data), now - 10),        # Requeue with the same job ID.
            (json.dumps(data), None),            # Scheduler cannot confirm the start.
            (json.dumps(dict(data, sampled_at=now - 91)), now - 150),
            (json.dumps(dict(data, sampled_at=now + 1000)), now - 150),
            (json.dumps(dict(data, count=-1)), now - 150),
            (json.dumps(dict(data, count=0)), now - 150),  # No devices cannot hold 512 MiB.
            ('{"count":', now - 150),
        ]
        for content, started in cases:
            with self.subTest(content=content, started=started):
                self.sample.write_text(content)
                usage = self.snapshot(started, now=now).gpu
                self.assertIsNone(usage.count)
                self.assertIsNone(usage.memory_bytes)
                self.assertTrue(usage.problem)

    def test_publication_failure_preserves_previous_file_and_does_not_interrupt_md(self):
        from mains import _gpu_monitor
        from mains._gpu import GpuUsage

        attempted = threading.Event()
        self.sample.write_text("previous sample")

        def failed_replace(source, destination):
            attempted.set()
            raise OSError("shared filesystem unavailable")

        with (
            mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_JOB_NAME": "solution-md-replica"}),
            mock.patch.object(_gpu_monitor, "measure_gpu", return_value=GpuUsage(1, 1024)),
            mock.patch.object(_gpu_monitor.os, "replace", side_effect=failed_replace),
            self.assertLogs("mains._gpu_monitor", level="WARNING") as logs,
            _gpu_monitor.monitor_gpu(self.logs, "solution", "replica"),
        ):
            self.assertTrue(attempted.wait(timeout=3))
        self.assertIn("shared filesystem unavailable", logs.output[0])
        self.assertEqual(self.sample.read_text(), "previous sample")
        self.assertEqual(list(self.logs.iterdir()), [self.sample])

    def test_gpu_monitor_failure_preserves_the_workflow_result_and_exception(self):
        from mains import _gpu_monitor

        real_error = RuntimeError("checkpoint validation failed")
        with (
            mock.patch.dict(os.environ, {"SLURM_JOB_NAME": ""}),
            mock.patch.object(threading.Thread, "start", side_effect=RuntimeError("no threads available")),
            self.assertLogs("mains._gpu_monitor", level="WARNING"),
        ):
            with _gpu_monitor.monitor_gpu(self.logs, "solution", "replica"):
                self.assertEqual(list(self.logs.iterdir()), [])
            with self.assertRaises(RuntimeError) as error, _gpu_monitor.monitor_gpu(self.logs, "solution", "replica"):
                raise real_error
        self.assertIs(error.exception, real_error)


if __name__ == "__main__":
    unittest.main()
