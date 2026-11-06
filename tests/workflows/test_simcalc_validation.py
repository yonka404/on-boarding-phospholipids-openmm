import argparse
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mains import _common
from mains._simcalc import validate_solution


class SimcalcValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.environment = {
            "SLURM_JOB_ID": "203",
            "CUDA_VISIBLE_DEVICES": "GPU-1234-abcd",
        }

    def run_validation(self, system_id: str = "replica_2") -> str:
        with (
            contextlib.chdir(self.root),
            mock.patch.object(_common, "PROJECT_ROOT", self.root),
            mock.patch.dict(os.environ, self.environment),
            mock.patch("importlib.metadata.version", return_value="8.5.2"),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            validate_solution.validate_solution(system_id)
        return output.getvalue()

    def test_success_reports_allocation_without_creating_outputs(self) -> None:
        output = self.run_validation()
        self.assertIn("Validated system=replica_2 job=203", output)
        for label in ("Allocation", "Environment", "Platform", "Output"):
            self.assertIn(label, output)
        self.assertIn("requested_platform=CUDA", output)
        self.assertIn("GPU-1234-abcd", output)
        self.assertIn("Python=", output)
        self.assertIn("OpenMM=8.5.2", output)
        self.assertIn(str(self.root / "data/outputs/openmm_native/solution/replica_2"), output)
        self.assertFalse((self.root / "data").exists())

    def test_allocation_check_does_not_validate_inputs_or_initialize_cuda(self) -> None:
        with (
            mock.patch(
                "charmm_gui_md.shared.inputs.openmm_native_files.OpenMMNativeFiles.from_root",
                side_effect=AssertionError("input validation belongs to StageRunner"),
            ),
            mock.patch(
                "openmm.Platform.getPlatformByName",
                side_effect=AssertionError("CUDA selection belongs to the simulation factory"),
            ),
            mock.patch(
                "openmm.Context",
                side_effect=AssertionError("CUDA initialization belongs to the simulation factory"),
            ),
        ):
            self.run_validation()

    def test_missing_input_folder_is_left_to_the_simulation_workflow(self) -> None:
        self.assertFalse((self.root / "data/inputs").exists())
        self.run_validation()

    def test_invalid_system_id_is_rejected(self) -> None:
        for system_id in ("", "..", "../other_replica"):
            with self.subTest(system_id=system_id), self.assertRaises(argparse.ArgumentTypeError):
                self.run_validation(system_id)

    def test_missing_or_invalid_job_id_is_rejected(self) -> None:
        for job_id in ("", "not-a-job"):
            with self.subTest(job_id=job_id):
                self.environment["SLURM_JOB_ID"] = job_id
                with self.assertRaisesRegex(ValueError, "SLURM_JOB_ID"):
                    self.run_validation()

    def test_missing_or_multiple_visible_gpus_are_rejected(self) -> None:
        for mask in ("", "-1", "0,1", "NoDevFiles"):
            with self.subTest(mask=mask):
                self.environment["CUDA_VISIBLE_DEVICES"] = mask
                with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                    self.run_validation()

    def test_supported_single_gpu_masks_are_reported_unchanged(self) -> None:
        for mask in ("2", "GPU-1234-abcd", "MIG-GPU-1234/1/0"):
            with self.subTest(mask=mask):
                self.environment["CUDA_VISIBLE_DEVICES"] = mask
                with mock.patch.dict(os.environ, self.environment):
                    self.assertIn(f"Slurm GPU {mask}", self.run_validation())
                    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], mask)

    def test_old_python_fails_with_environment_instructions(self) -> None:
        with (
            mock.patch.object(validate_solution.sys, "version_info", (3, 8, 10)),
            self.assertRaisesRegex(RuntimeError, "Python 3.12.10.*.venv"),
        ):
            self.run_validation()

    def test_main_returns_clear_error_without_traceback(self) -> None:
        with (
            mock.patch.object(validate_solution, "validate_solution", side_effect=ValueError("bad GPU mask")),
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            result = validate_solution.main(["replica_2"])
        self.assertEqual(result, 1)
        self.assertIn("bad GPU mask", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
