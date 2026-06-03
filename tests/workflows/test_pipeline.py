import unittest
from pathlib import Path
from unittest import mock

from charmm_gui_md.membrane import pipeline as membrane_pipeline
from charmm_gui_md.membrane.profile import MEMBRANE_PROFILE
from charmm_gui_md.solution import pipeline as solution_pipeline
from charmm_gui_md.solution.profile import SOLUTION_PROFILE


class PipelineTests(unittest.TestCase):
    def test_membrane_single_step_binds_membrane_profile(self) -> None:
        result = Path("outputs/step6.1_equilibration/final_coordinates.pdb")

        with mock.patch(
            "charmm_gui_md.membrane.pipeline.StageRunner",
        ) as runner_type:
            runner_type.return_value.run.return_value = result

            actual = membrane_pipeline.run_single_step(
                inputs_dir=Path("inputs"),
                outputs_dir=Path("outputs"),
                step_name="step6.1_equilibration",
            )

        self.assertEqual(actual, result)
        runner_type.assert_called_once_with(profile=MEMBRANE_PROFILE)

    def test_solution_single_step_binds_solution_profile(self) -> None:
        result = Path("outputs/step4_equilibration/final_coordinates.pdb")

        with mock.patch(
            "charmm_gui_md.solution.pipeline.StageRunner",
        ) as runner_type:
            runner_type.return_value.run.return_value = result

            actual = solution_pipeline.run_single_step(
                inputs_dir=Path("inputs"),
                outputs_dir=Path("outputs"),
                step_name="step4_equilibration",
            )

        self.assertEqual(actual, result)
        runner_type.assert_called_once_with(profile=SOLUTION_PROFILE)

    def test_solution_protocol_sweep_binds_solution_profile(self) -> None:
        with mock.patch(
            "charmm_gui_md.solution.pipeline.SweepRunner",
        ) as runner_type:
            solution_pipeline.run_protocol_sweep(
                inputs_dir=Path("inputs"),
                outputs_dir=Path("outputs"),
            )

        runner_type.assert_called_once_with(profile=SOLUTION_PROFILE)


if __name__ == "__main__":
    unittest.main()
