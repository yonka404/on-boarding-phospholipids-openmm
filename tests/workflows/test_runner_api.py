import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from charmm_gui_md.shared.artifacts import StageArtifacts
from charmm_gui_md.shared.workflows import StageRunner, SweepRunner
from charmm_gui_md.solution.profile import SOLUTION_PROFILE


class RunnerApiTests(unittest.TestCase):
    def test_shared_runners_require_an_explicit_profile(self) -> None:
        stage_parameters = inspect.signature(StageRunner).parameters
        sweep_parameters = inspect.signature(SweepRunner).parameters

        self.assertIs(stage_parameters["profile"].default, inspect.Parameter.empty)
        self.assertIs(sweep_parameters["profile"].default, inspect.Parameter.empty)

    def test_stage_runner_rejects_stage_outside_selected_profile(self) -> None:
        runner = StageRunner(profile=SOLUTION_PROFILE)

        with self.assertRaisesRegex(ValueError, r"step6\.1_equilibration"):
            runner.run(
                inputs_dir=Path("inputs"),
                outputs_dir=Path("outputs"),
                step_name="step6.1_equilibration",
            )

    def test_solution_sweep_runs_solution_stages_in_order(self) -> None:
        stage_runner = mock.Mock()
        runner = SweepRunner(
            profile=SOLUTION_PROFILE,
            stage_runner=stage_runner,
        )
        inputs_dir = Path("inputs")
        outputs_dir = Path("outputs")

        runner.run(inputs_dir=inputs_dir, outputs_dir=outputs_dir)

        self.assertEqual(
            stage_runner.run.call_args_list,
            [
                mock.call(
                    inputs_dir=inputs_dir,
                    outputs_dir=outputs_dir,
                    step_name="step4_equilibration",
                ),
                mock.call(
                    inputs_dir=inputs_dir,
                    outputs_dir=outputs_dir,
                    step_name="step5_production",
                ),
            ],
        )

    def test_solution_sweep_skips_completed_stage_before_interrupted_stage(self) -> None:
        stage_runner = mock.Mock()
        runner = SweepRunner(
            profile=SOLUTION_PROFILE,
            stage_runner=stage_runner,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            completed = StageArtifacts.for_stage(outputs_dir, "step4_equilibration")
            completed.create_directory()
            completed.final_state_path.write_text("<state />")
            completed.final_coordinates_path.write_text("ATOM")

            interrupted = StageArtifacts.for_stage(outputs_dir, "step5_production")
            interrupted.create_directory()
            interrupted.checkpoint_path.write_bytes(b"checkpoint")

            runner.run(inputs_dir=Path("inputs"), outputs_dir=outputs_dir)

        stage_runner.run.assert_called_once_with(
            inputs_dir=Path("inputs"),
            outputs_dir=outputs_dir,
            step_name="step5_production",
        )

    def test_sweep_rejects_out_of_order_results_before_running_any_stage(self) -> None:
        stage_runner = mock.Mock()
        runner = SweepRunner(profile=SOLUTION_PROFILE, stage_runner=stage_runner)
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir)
            later = StageArtifacts.for_stage(outputs, "step5_production")
            later.create_directory()
            later.final_state_path.write_text("<state />")
            later.final_coordinates_path.write_text("ATOM")
            with self.assertRaisesRegex(RuntimeError, "stage order"):
                runner.run(Path("inputs"), outputs)
        stage_runner.run.assert_not_called()

    def test_sweep_rejects_empty_stage_before_running(self) -> None:
        stage_runner = mock.Mock()
        runner = SweepRunner(profile=SOLUTION_PROFILE, stage_runner=stage_runner)
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir)
            StageArtifacts.for_stage(outputs, "step4_equilibration").create_directory()
            with self.assertRaisesRegex(RuntimeError, "incomplete.*checkpoint"):
                runner.run(Path("inputs"), outputs)
        stage_runner.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
