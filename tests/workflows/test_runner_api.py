import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from charmm_gui_md.membrane.profile import MEMBRANE_PROFILE
from charmm_gui_md.shared.artifacts import StageArtifacts
from charmm_gui_md.shared.workflows import StageRunner, SweepRunner
from charmm_gui_md.solution.profile import SOLUTION_PROFILE

_PROTOCOLS = (
    (
        MEMBRANE_PROFILE,
        (
            "step6.1_equilibration",
            "step6.2_equilibration",
            "step6.3_equilibration",
            "step6.4_equilibration",
            "step6.5_equilibration",
            "step6.6_equilibration",
            "step7_production",
        ),
    ),
    (SOLUTION_PROFILE, ("step4_equilibration", "step5_production")),
)


class RunnerApiTests(unittest.TestCase):
    def test_shared_runners_require_an_explicit_profile(self) -> None:
        stage_parameters = inspect.signature(StageRunner).parameters
        sweep_parameters = inspect.signature(SweepRunner).parameters

        self.assertIs(stage_parameters["profile"].default, inspect.Parameter.empty)
        self.assertIs(sweep_parameters["profile"].default, inspect.Parameter.empty)

    def test_stage_runner_rejects_stage_outside_each_selected_profile(self) -> None:
        for profile, invalid_stage in (
            (SOLUTION_PROFILE, "step6.1_equilibration"),
            (MEMBRANE_PROFILE, "step5_production"),
        ):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmpdir:
                outputs = Path(tmpdir) / "outputs"
                with self.assertRaises(ValueError) as error:
                    StageRunner(profile=profile).run(
                        Path(tmpdir) / "inputs", outputs, invalid_stage
                    )
                self.assertIn(invalid_stage, str(error.exception))
                self.assertFalse(outputs.exists())

    def test_both_sweeps_run_stages_in_order(self) -> None:
        for profile, stages in _PROTOCOLS:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmpdir:
                stage_runner = mock.Mock()
                inputs = Path(tmpdir) / "inputs"
                outputs = Path(tmpdir) / "outputs"

                SweepRunner(profile=profile, stage_runner=stage_runner).run(
                    inputs, outputs
                )

                self.assertEqual(
                    stage_runner.run.call_args_list,
                    [
                        mock.call(
                            inputs_dir=inputs, outputs_dir=outputs, step_name=name
                        )
                        for name in stages
                    ],
                )
                self.assertFalse(
                    outputs.exists(), "Only the stage runner should create outputs"
                )

    def test_both_sweeps_propagate_stage_failure_and_stop_before_later_stages(
        self,
    ) -> None:
        for profile, stages in _PROTOCOLS:
            for failed_index, failed_stage in enumerate(stages):
                with (
                    self.subTest(profile=profile, failed_stage=failed_stage),
                    tempfile.TemporaryDirectory() as tmpdir,
                ):
                    failure = RuntimeError(f"{failed_stage} failed")
                    stage_runner = mock.Mock()
                    stage_runner.run.side_effect = [None] * failed_index + [failure]
                    inputs = Path(tmpdir) / "inputs"
                    outputs = Path(tmpdir) / "outputs"

                    with self.assertRaises(RuntimeError) as error:
                        SweepRunner(profile=profile, stage_runner=stage_runner).run(
                            inputs, outputs
                        )

                    self.assertIs(error.exception, failure)
                    self.assertEqual(
                        stage_runner.run.call_args_list,
                        [
                            mock.call(
                                inputs_dir=inputs, outputs_dir=outputs, step_name=name
                            )
                            for name in stages[: failed_index + 1]
                        ],
                    )

    def test_both_sweeps_skip_completed_stages_before_interrupted_stage(self) -> None:
        for profile, stages in _PROTOCOLS:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmpdir:
                outputs = Path(tmpdir)
                stage_runner = mock.Mock()
                for name in stages[:-1]:
                    self._complete_stage(outputs, name)
                interrupted = StageArtifacts.for_stage(outputs, stages[-1])
                interrupted.create_directory()
                interrupted.checkpoint_path.write_bytes(b"checkpoint")

                SweepRunner(profile=profile, stage_runner=stage_runner).run(
                    Path("unused"), outputs
                )

                stage_runner.run.assert_called_once_with(
                    inputs_dir=Path("unused"), outputs_dir=outputs, step_name=stages[-1]
                )

    def test_both_completed_sweeps_do_not_run_any_stage(self) -> None:
        for profile, stages in _PROTOCOLS:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmpdir:
                outputs = Path(tmpdir)
                for name in stages:
                    self._complete_stage(outputs, name)
                stage_runner = mock.Mock()

                SweepRunner(profile=profile, stage_runner=stage_runner).run(
                    Path("unused"), outputs
                )

                stage_runner.run.assert_not_called()

    def test_sweep_rejects_out_of_order_results_before_running_any_stage(self) -> None:
        for profile, stages in _PROTOCOLS:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmpdir:
                outputs = Path(tmpdir)
                self._complete_stage(outputs, stages[1])
                stage_runner = mock.Mock()
                with self.assertRaisesRegex(RuntimeError, "stage order"):
                    SweepRunner(profile=profile, stage_runner=stage_runner).run(
                        Path("unused"), outputs
                    )
                stage_runner.run.assert_not_called()

    def test_sweep_rejects_empty_stage_before_running(self) -> None:
        for profile, stages in _PROTOCOLS:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmpdir:
                outputs = Path(tmpdir)
                StageArtifacts.for_stage(outputs, stages[0]).create_directory()
                stage_runner = mock.Mock()
                with self.assertRaisesRegex(RuntimeError, "incomplete.*checkpoint"):
                    SweepRunner(profile=profile, stage_runner=stage_runner).run(
                        Path("unused"), outputs
                    )
                stage_runner.run.assert_not_called()

    @staticmethod
    def _complete_stage(outputs: Path, name: str) -> None:
        artifacts = StageArtifacts.for_stage(outputs, name)
        artifacts.create_directory()
        artifacts.final_state_path.write_text("<state />")
        artifacts.final_coordinates_path.write_text("ATOM")


if __name__ == "__main__":
    unittest.main()
