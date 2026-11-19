import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mains import _gpu


class GpuUsageTests(unittest.TestCase):
    pid = 42

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.kfd = Path(directory.name) / "kfd" / "proc"
        kfd_path = mock.patch.object(_gpu, "_KFD_PROC", self.kfd)
        kfd_path.start()
        self.addCleanup(kfd_path.stop)
        nvidia = mock.patch.object(
            _gpu.subprocess, "run", side_effect=FileNotFoundError("nvidia-smi")
        )
        self.run_nvidia = nvidia.start()
        self.addCleanup(nvidia.stop)

    def nvidia_reports(self, stdout, returncode=0):
        def query(args, **kwargs):
            self.assertEqual(list(args), [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ])
            self.assertGreater(kwargs["timeout"], 0)
            self.assertLessEqual(kwargs["timeout"], 5)
            return subprocess.CompletedProcess(args, returncode, stdout, "")

        self.run_nvidia.side_effect = query

    def kfd_field(self, relative_path, value, pid=42):
        path = self.kfd / str(pid) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(value), encoding="ascii")

    def queue(self, queue_id, gpu_id, kind=0):
        self.kfd_field(f"queues/{queue_id}/type", kind)
        self.kfd_field(f"queues/{queue_id}/gpuid", gpu_id)

    def test_nvidia_sums_distinct_devices_for_the_exact_pid(self):
        self.nvidia_reports(
            "GPU-aaa, 424, 500\nGPU-aaa, 42, 256\nGPU-aaa, 42, 256\n"
            "GPU-bbb, 42, 512\nGPU-ccc, 4242, [N/A]\n"
        )
        self.assertEqual(_gpu.measure_gpu(self.pid), _gpu.GpuUsage(2, 805306368))

    def test_nvidia_preserves_device_count_when_memory_is_unavailable(self):
        for last_row in ("GPU-bbb,42", "GPU-bbb,42,[N/A]", "GPU-bbb,42,-1",
                         "GPU-bbb,42,3\nGPU-bbb,42,4"):
            with self.subTest(last_row=last_row):
                self.nvidia_reports("GPU-aaa,42,100\n" + last_row)
                usage = _gpu.measure_gpu(self.pid)
                self.assertEqual(usage.count, 2)
                self.assertIsNone(usage.memory_bytes)
                self.assertIn("memory", (usage.problem or "").lower())

    def test_unattributable_nvidia_rows_do_not_create_measurements(self):
        for row in ("[N/A],42,32", "GPU-aaa,unknown,32", "unidentified processes"):
            with self.subTest(row=row):
                self.nvidia_reports(row)
                usage = _gpu.measure_gpu(self.pid)
                self.assertIsNone(usage.count)
                self.assertIsNone(usage.memory_bytes)
                self.assertTrue(usage.problem)

    def test_failed_nvidia_queries_ignore_partial_stdout(self):
        self.nvidia_reports("GPU-aaa,42,512", returncode=15)
        usage = _gpu.measure_gpu(self.pid)
        self.assertIsNone(usage.count)
        self.assertIsNone(usage.memory_bytes)
        self.assertIn("NVIDIA", usage.problem or "")

    def test_nvidia_timeout_has_an_explanation(self):
        self.run_nvidia.side_effect = subprocess.TimeoutExpired("nvidia-smi", 2)
        usage = _gpu.measure_gpu(self.pid)
        self.assertIsNone(usage.count)
        self.assertIn("timed out", usage.problem or "")

    def test_missing_backends_and_unlisted_pids_are_unknown(self):
        usage = _gpu.measure_gpu(self.pid)
        self.assertEqual((usage.count, usage.memory_bytes), (None, None))
        self.assertTrue(usage.problem)

        self.kfd_field("vram_100", 500, pid=142)
        self.nvidia_reports("GPU-aaa,142,50")
        usage = _gpu.measure_gpu(self.pid)
        self.assertEqual((usage.count, usage.memory_bytes), (None, None))
        self.assertIn("PID", usage.problem or "")

    def test_kfd_counts_devices_with_vram_or_compute_queues_once(self):
        self.kfd_field("vram_100", 700)
        self.kfd_field("vram_200", 0)
        self.kfd_field("vram_300", 0)
        self.kfd_field("vram_400", 300)
        self.kfd_field("vram_100", 9000, pid=142)
        self.queue(0, 100)
        self.queue(1, 200)
        self.queue(2, 200)
        self.queue(3, 300, kind=1)  # DMA alone does not attest to compute use.
        self.assertEqual(_gpu.measure_gpu(self.pid), _gpu.GpuUsage(3, 1000))

    def test_idle_kfd_process_with_explicit_counters_can_report_zero(self):
        self.kfd_field("vram_100", 0)
        (self.kfd / str(self.pid) / "queues").mkdir()
        self.assertEqual(_gpu.measure_gpu(self.pid), _gpu.GpuUsage(0, 0))

    def test_kfd_keeps_gpu_count_when_its_vram_counter_is_unreadable(self):
        self.kfd_field("vram_100", "malformed")
        self.queue(0, 100)
        usage = _gpu.measure_gpu(self.pid)
        self.assertEqual((usage.count, usage.memory_bytes), (1, None))
        self.assertIn("VRAM", usage.problem or "")

    def test_missing_vram_counter_for_a_queue_prevents_a_partial_memory_total(self):
        self.kfd_field("vram_100", 700)
        self.queue(0, 200)
        usage = _gpu.measure_gpu(self.pid)
        self.assertEqual((usage.count, usage.memory_bytes), (2, None))
        self.assertTrue(usage.problem)

    def test_disappearing_queue_keeps_memory_but_cannot_establish_count(self):
        self.kfd_field("vram_100", 128)
        self.kfd_field("vram_200", 0)
        self.kfd_field("queues/0/type", 0)
        usage = _gpu.measure_gpu(self.pid)
        self.assertEqual((usage.count, usage.memory_bytes), (None, 128))
        self.assertTrue(usage.problem)

    def test_empty_nvidia_list_does_not_hide_an_amd_process(self):
        self.nvidia_reports("")
        self.kfd_field("vram_100", 256)
        self.queue(0, 100)
        self.assertEqual(_gpu.measure_gpu(self.pid), _gpu.GpuUsage(1, 256))

    def test_an_unloaded_nvidia_driver_does_not_hide_an_amd_reading(self):
        self.nvidia_reports("", returncode=9)
        self.kfd_field("vram_100", 256)
        self.queue(0, 100)
        usage = _gpu.measure_gpu(self.pid)
        self.assertEqual(usage, _gpu.GpuUsage(1, 256))

    def test_failed_query_on_a_mixed_node_cannot_return_an_exact_total(self):
        self.nvidia_reports("", returncode=15)
        self.kfd_field("vram_100", 256)
        self.queue(0, 100)
        usage = _gpu.measure_gpu(self.pid)
        self.assertEqual((usage.count, usage.memory_bytes), (None, None))
        self.assertIn("NVIDIA", usage.problem or "")

    def test_a_missing_kfd_memory_field_does_not_hide_known_device_counts(self):
        self.nvidia_reports("GPU-aaa,42,1")
        self.queue(0, 100)
        usage = _gpu.measure_gpu(self.pid)
        self.assertEqual((usage.count, usage.memory_bytes), (2, None))
        self.assertIn("VRAM", usage.problem or "")

    def test_combines_pid_records_from_both_drivers(self):
        self.nvidia_reports("GPU-aaa,42,1")
        self.kfd_field("vram_100", 256)
        self.queue(0, 100)
        self.assertEqual(_gpu.measure_gpu(self.pid), _gpu.GpuUsage(2, 1048832))
