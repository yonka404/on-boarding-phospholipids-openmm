import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from charmm_gui_md.shared.artifacts import StageArtifacts
from charmm_gui_md.shared.simulation.checkpoints import StageCheckpointManager


class StageCheckpointManagerTests(unittest.TestCase):
    def test_restore_returns_none_when_no_checkpoint_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = StageArtifacts.for_stage(Path(tmpdir), "step5_production")
            simulation = mock.Mock()

            remaining_steps = StageCheckpointManager().restore(
                simulation,
                artifacts,
                SimpleNamespace(
                    step_name="step5_production",
                    dynamics_steps=100,
                ),
            )

        self.assertIsNone(remaining_steps)
        simulation.loadCheckpoint.assert_not_called()

    def test_restore_loads_checkpoint_and_returns_remaining_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = StageArtifacts.for_stage(Path(tmpdir), "step5_production")
            artifacts.create_directory()
            artifacts.checkpoint_path.write_bytes(b"checkpoint")
            simulation = mock.Mock(currentStep=40)

            remaining_steps = StageCheckpointManager().restore(
                simulation,
                artifacts,
                SimpleNamespace(
                    step_name="step5_production",
                    dynamics_steps=100,
                ),
            )

        simulation.loadCheckpoint.assert_called_once_with(
            str(artifacts.checkpoint_path),
        )
        self.assertEqual(remaining_steps, 60)

    def test_restore_rejects_checkpoint_beyond_stage_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = StageArtifacts.for_stage(Path(tmpdir), "step5_production")
            artifacts.create_directory()
            artifacts.checkpoint_path.write_bytes(b"checkpoint")
            simulation = mock.Mock(currentStep=101)

            with self.assertRaisesRegex(
                RuntimeError,
                r"checkpoint step 101 exceeds configured stage target 100",
            ):
                StageCheckpointManager().restore(
                    simulation,
                    artifacts,
                    SimpleNamespace(
                        step_name="step5_production",
                        dynamics_steps=100,
                    ),
                )

    def test_restore_rejects_negative_checkpoint_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = StageArtifacts.for_stage(Path(tmpdir), "step5_production")
            artifacts.create_directory()
            artifacts.checkpoint_path.write_bytes(b"checkpoint")
            simulation = mock.Mock(currentStep=-1)

            with self.assertRaisesRegex(
                RuntimeError,
                r"checkpoint step -1 is invalid",
            ):
                StageCheckpointManager().restore(
                    simulation,
                    artifacts,
                    SimpleNamespace(
                        step_name="step5_production",
                        dynamics_steps=100,
                    ),
                )

    def test_restore_preserves_checkpoint_when_openmm_rejects_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = StageArtifacts.for_stage(Path(tmpdir), "step5_production")
            artifacts.create_directory()
            artifacts.checkpoint_path.write_bytes(b"checkpoint")
            simulation = mock.Mock()
            simulation.loadCheckpoint.side_effect = ValueError("incompatible")

            with self.assertRaisesRegex(
                RuntimeError,
                r"Could not load checkpoint.*checkpoint\.chk",
            ):
                StageCheckpointManager().restore(
                    simulation,
                    artifacts,
                    SimpleNamespace(
                        step_name="step5_production",
                        dynamics_steps=100,
                    ),
                )

            self.assertTrue(artifacts.checkpoint_path.is_file())

    def test_clear_removes_completed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = StageArtifacts.for_stage(Path(tmpdir), "step5_production")
            artifacts.create_directory()
            artifacts.checkpoint_path.write_bytes(b"checkpoint")

            StageCheckpointManager().clear(
                artifacts,
                SimpleNamespace(step_name="step5_production"),
            )

            self.assertFalse(artifacts.checkpoint_path.exists())


if __name__ == "__main__":
    unittest.main()
