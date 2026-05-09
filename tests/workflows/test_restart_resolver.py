import tempfile
import unittest
from pathlib import Path

from protein_membrane_md.artifacts import RestartResolver, StageArtifacts

class RestartResolverTests(unittest.TestCase):
    def test_first_stage_uses_input_adapter_initial_coordinates(self) -> None:
        resolver = RestartResolver()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            initial_coordinates = root / "openmm" / "step5_input.pdb"

            restart_source = resolver.resolve(
                inputs_dir=root / "openmm",
                outputs_dir=root / "outputs",
                step_name="step6.1_equilibration",
                initial_coordinates_path=initial_coordinates,
            )

        self.assertEqual(restart_source.coordinates_path, initial_coordinates)
        self.assertIsNone(restart_source.state_path)
        self.assertEqual(restart_source.description, "initial input coordinates")

    def test_later_stage_prefers_previous_state_when_available(self) -> None:
        resolver = RestartResolver()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outputs_dir = root / "outputs"
            previous_artifacts = StageArtifacts.for_stage(
                outputs_dir,
                "step6.1_equilibration",
            )
            previous_artifacts.create_directory()
            previous_artifacts.final_state_path.write_text("<state />")

            restart_source = resolver.resolve(
                inputs_dir=root / "openmm",
                outputs_dir=outputs_dir,
                step_name="step6.2_equilibration",
                initial_coordinates_path=root / "openmm" / "step5_input.pdb",
            )

        self.assertEqual(
            restart_source.coordinates_path,
            previous_artifacts.final_coordinates_path,
        )
        self.assertEqual(restart_source.state_path, previous_artifacts.final_state_path)
        self.assertEqual(
            restart_source.description,
            "restart from previous protocol stage 'step6.1_equilibration'",
        )


if __name__ == "__main__":
    unittest.main()
