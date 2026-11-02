import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from charmm_gui_md import membrane, solution
from mains import local

PROJECT = Path(__file__).resolve().parents[2]


class MainScriptPathTests(unittest.TestCase):
    def test_full_protocol_dispatches_to_selected_environment(self) -> None:
        for environment, pipeline in (("membrane", membrane), ("solution", solution)):
            with (
                self.subTest(environment=environment),
                mock.patch.object(pipeline, "run_protocol_sweep") as run_sweep,
                mock.patch.object(pipeline, "run_single_step") as run_step,
            ):
                local.main([environment, "replica_2"])

                run_sweep.assert_called_once_with(
                    inputs_dir=PROJECT / "data/inputs/openmm_native" / environment / "replica_2",
                    outputs_dir=PROJECT / "data/outputs/openmm_native" / environment / "replica_2",
                )
                run_step.assert_not_called()

    def test_single_stage_dispatches_to_selected_environment(self) -> None:
        for environment, pipeline, stage in (
            ("membrane", membrane, "step6.3_equilibration"),
            ("solution", solution, "step5_production"),
        ):
            with (
                self.subTest(environment=environment),
                mock.patch.object(pipeline, "run_single_step") as run_step,
                mock.patch.object(pipeline, "run_protocol_sweep") as run_sweep,
            ):
                local.main([environment, "replica_2", "--stage", stage])

                run_step.assert_called_once_with(
                    inputs_dir=PROJECT / "data/inputs/openmm_native" / environment / "replica_2",
                    outputs_dir=PROJECT / "data/outputs/openmm_native" / environment / "replica_2",
                    step_name=stage,
                )
                run_sweep.assert_not_called()

    def test_paths_are_independent_of_working_directory(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.chdir(directory),
            mock.patch.object(solution, "run_protocol_sweep") as run_sweep,
        ):
            local.main(["solution", "abeta_40"])

            run_sweep.assert_called_once_with(
                inputs_dir=PROJECT / "data/inputs/openmm_native/solution/abeta_40",
                outputs_dir=PROJECT / "data/outputs/openmm_native/solution/abeta_40",
            )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_local_system_id_keeps_existing_whitespace_rules(self) -> None:
        with mock.patch.object(solution, "run_protocol_sweep") as run_sweep:
            local.main(["solution", " replica 2 "])

        self.assertEqual(
            run_sweep.call_args.kwargs["inputs_dir"],
            PROJECT / "data/inputs/openmm_native/solution/replica 2",
        )

    def test_invalid_arguments_never_start_a_simulation(self) -> None:
        invalid_arguments = [
            [],
            ["unknown", "replica"],
            ["solution", "replica", "--stage"],
            ["solution", "replica", "--stage", "step6.1_equilibration"],
            ["membrane", "replica", "--stage", "step5_production"],
            ["solution", "replica", "--stage", "unknown"],
            ["solution", "replica", "--unknown"],
        ]
        invalid_arguments.extend(
            ["solution", system_id]
            for system_id in ("", " ", ".", "..", "../bad", "a/b", "a\\b")
        )
        with (
            mock.patch.object(solution, "run_protocol_sweep") as solution_sweep,
            mock.patch.object(solution, "run_single_step") as solution_step,
            mock.patch.object(membrane, "run_protocol_sweep") as membrane_sweep,
            mock.patch.object(membrane, "run_single_step") as membrane_step,
        ):
            for arguments in invalid_arguments:
                with (
                    self.subTest(arguments=arguments),
                    contextlib.redirect_stderr(io.StringIO()) as errors,
                    self.assertRaises(SystemExit) as failure,
                ):
                    local.main(arguments)
                self.assertEqual(failure.exception.code, 2)
                self.assertIn("error:", errors.getvalue())

            for run in (solution_sweep, solution_step, membrane_sweep, membrane_step):
                run.assert_not_called()

    def test_invalid_stage_lists_the_selected_protocol_stages(self) -> None:
        with (
            contextlib.redirect_stderr(io.StringIO()) as errors,
            self.assertRaises(SystemExit),
        ):
            local.main(["solution", "replica", "--stage", "step6.1_equilibration"])

        for stage in solution.PROTOCOL_STAGES:
            self.assertIn(stage, errors.getvalue())

    def test_direct_script_help_works_outside_repository_without_site_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-S", str(PROJECT / "mains/local.py"), "--help"],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("solution", result.stdout)
        self.assertIn("membrane", result.stdout)
        self.assertIn("--stage", result.stdout)


if __name__ == "__main__":
    unittest.main()
