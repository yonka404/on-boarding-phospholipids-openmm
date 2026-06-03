import inspect
import unittest
from pathlib import Path
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
