import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from charmm_gui_md.shared.artifacts import StageArtifacts
from charmm_gui_md.shared.workflows.stage_runner import StageRunner


class StageRunnerCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = mock.Mock()
        self.restart_resolver = mock.Mock()
        self.simulation_factory = mock.Mock()
        self.simulation_initializer = mock.Mock()
        self.reporter_installer = mock.Mock()
        self.output_writer = mock.Mock()
        self.checkpoint_manager = mock.Mock()
        self.simulation = mock.Mock(currentStep=40)
        self.simulation_factory.create.return_value = self.simulation
        self.files = SimpleNamespace(
            inputs_dir=Path("inputs/openmm"),
            initial_coordinates_path=Path("inputs/openmm/step3_input.pdb"),
        )
        self.protocol = SimpleNamespace(
            step_name="step5_production",
            dynamics_steps=100,
            timestep_ps=0.002,
            has_minimization=True,
            minimization_tolerance_kj_mol_nm=10.0,
            minimization_steps=500,
        )

    def _runner(self) -> StageRunner:
        return StageRunner(
            profile=self.profile,
            restart_resolver=self.restart_resolver,
            simulation_factory=self.simulation_factory,
            simulation_initializer=self.simulation_initializer,
            reporter_installer=self.reporter_installer,
            output_writer=self.output_writer,
            checkpoint_manager=self.checkpoint_manager,
        )

    def test_resume_skips_initialization_and_runs_only_remaining_steps(self) -> None:
        self.checkpoint_manager.restore.return_value = 60

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            artifacts = StageArtifacts.for_stage(outputs_dir, "step5_production")
            expected_result = artifacts.final_coordinates_path
            self.output_writer.write.return_value = expected_result

            with (
                mock.patch(
                    "charmm_gui_md.shared.workflows.stage_runner.OpenMMNativeFiles.from_root",
                    return_value=self.files,
                ),
                mock.patch(
                    "charmm_gui_md.shared.workflows.stage_runner.OpenMMStageProtocol.from_file",
                    return_value=self.protocol,
                ),
            ):
                result = self._runner().run(
                    inputs_dir=Path("inputs"),
                    outputs_dir=outputs_dir,
                    step_name="step5_production",
                )

        self.simulation_initializer.initialize.assert_not_called()
        self.simulation_initializer.initialize_velocities.assert_not_called()
        self.restart_resolver.resolve.assert_not_called()
        self.simulation.minimizeEnergy.assert_not_called()
        self.reporter_installer.install.assert_called_once_with(
            self.simulation,
            artifacts,
            self.protocol,
            append=True,
        )
        self.simulation.step.assert_called_once_with(60)
        self.checkpoint_manager.clear.assert_called_once_with(
            artifacts,
            self.protocol,
        )
        self.assertEqual(result, expected_result)

    def test_coordinate_start_initializes_velocities_after_minimization(self) -> None:
        self.checkpoint_manager.restore.return_value = None
        restart_source = SimpleNamespace(state_path=None)
        self.restart_resolver.resolve.return_value = restart_source
        lifecycle_events: list[str] = []
        self.simulation_initializer.initialize.side_effect = (
            lambda *_: lifecycle_events.append("initialize")
        )
        self.simulation.minimizeEnergy.side_effect = (
            lambda **_: lifecycle_events.append("minimize")
        )
        self.simulation_initializer.initialize_velocities.side_effect = (
            lambda *_: lifecycle_events.append("initialize_velocities")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            artifacts = StageArtifacts.for_stage(outputs_dir, "step5_production")
            self.output_writer.write.return_value = artifacts.final_coordinates_path

            with (
                mock.patch(
                    "charmm_gui_md.shared.workflows.stage_runner.OpenMMNativeFiles.from_root",
                    return_value=self.files,
                ),
                mock.patch(
                    "charmm_gui_md.shared.workflows.stage_runner.OpenMMStageProtocol.from_file",
                    return_value=self.protocol,
                ),
            ):
                self._runner().run(
                    inputs_dir=Path("inputs"),
                    outputs_dir=outputs_dir,
                    step_name="step5_production",
                )

        self.simulation_initializer.initialize.assert_called_once_with(
            self.simulation,
            restart_source,
            self.protocol,
        )
        self.simulation_initializer.initialize_velocities.assert_called_once_with(
            self.simulation,
            self.protocol,
        )
        self.simulation.minimizeEnergy.assert_called_once()
        self.assertEqual(
            lifecycle_events,
            ["initialize", "minimize", "initialize_velocities"],
        )
        self.reporter_installer.install.assert_called_once_with(
            self.simulation,
            artifacts,
            self.protocol,
            append=False,
        )
        self.simulation.step.assert_called_once_with(100)

    def test_coordinate_start_without_minimization_initializes_velocities_before_dynamics(
        self,
    ) -> None:
        self.checkpoint_manager.restore.return_value = None
        self.protocol.has_minimization = False
        restart_source = SimpleNamespace(state_path=None)
        self.restart_resolver.resolve.return_value = restart_source
        lifecycle_events: list[str] = []
        self.simulation_initializer.initialize.side_effect = (
            lambda *_: lifecycle_events.append("initialize")
        )
        self.simulation_initializer.initialize_velocities.side_effect = (
            lambda *_: lifecycle_events.append("initialize_velocities")
        )
        self.simulation.step.side_effect = lambda *_: lifecycle_events.append("step")

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            artifacts = StageArtifacts.for_stage(outputs_dir, "step5_production")
            self.output_writer.write.return_value = artifacts.final_coordinates_path

            with (
                mock.patch(
                    "charmm_gui_md.shared.workflows.stage_runner.OpenMMNativeFiles.from_root",
                    return_value=self.files,
                ),
                mock.patch(
                    "charmm_gui_md.shared.workflows.stage_runner.OpenMMStageProtocol.from_file",
                    return_value=self.protocol,
                ),
            ):
                self._runner().run(
                    inputs_dir=Path("inputs"),
                    outputs_dir=outputs_dir,
                    step_name="step5_production",
                )

        self.simulation.minimizeEnergy.assert_not_called()
        self.assertEqual(
            lifecycle_events,
            ["initialize", "initialize_velocities", "step"],
        )

    def test_dynamic_state_start_preserves_loaded_velocities(self) -> None:
        self.checkpoint_manager.restore.return_value = None
        restart_source = SimpleNamespace(state_path=Path("previous_state.xml"))
        self.restart_resolver.resolve.return_value = restart_source

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            artifacts = StageArtifacts.for_stage(outputs_dir, "step5_production")
            self.output_writer.write.return_value = artifacts.final_coordinates_path

            with (
                mock.patch(
                    "charmm_gui_md.shared.workflows.stage_runner.OpenMMNativeFiles.from_root",
                    return_value=self.files,
                ),
                mock.patch(
                    "charmm_gui_md.shared.workflows.stage_runner.OpenMMStageProtocol.from_file",
                    return_value=self.protocol,
                ),
            ):
                self._runner().run(
                    inputs_dir=Path("inputs"),
                    outputs_dir=outputs_dir,
                    step_name="step5_production",
                )

        self.simulation_initializer.initialize.assert_called_once_with(
            self.simulation,
            restart_source,
            self.protocol,
        )
        self.simulation_initializer.initialize_velocities.assert_not_called()

    def test_completed_checkpoint_writes_outputs_without_zero_step_call(self) -> None:
        self.checkpoint_manager.restore.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            artifacts = StageArtifacts.for_stage(outputs_dir, "step5_production")
            self.output_writer.write.return_value = artifacts.final_coordinates_path

            with (
                mock.patch(
                    "charmm_gui_md.shared.workflows.stage_runner.OpenMMNativeFiles.from_root",
                    return_value=self.files,
                ),
                mock.patch(
                    "charmm_gui_md.shared.workflows.stage_runner.OpenMMStageProtocol.from_file",
                    return_value=self.protocol,
                ),
            ):
                self._runner().run(
                    inputs_dir=Path("inputs"),
                    outputs_dir=outputs_dir,
                    step_name="step5_production",
                )

        self.simulation.step.assert_not_called()
        self.output_writer.write.assert_called_once()
        self.checkpoint_manager.clear.assert_called_once()

    def test_checkpoint_is_retained_when_final_output_write_fails(self) -> None:
        self.checkpoint_manager.restore.return_value = 60
        self.output_writer.write.side_effect = OSError("disk full")

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch(
                    "charmm_gui_md.shared.workflows.stage_runner.OpenMMNativeFiles.from_root",
                    return_value=self.files,
                ),
                mock.patch(
                    "charmm_gui_md.shared.workflows.stage_runner.OpenMMStageProtocol.from_file",
                    return_value=self.protocol,
                ),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    self._runner().run(
                        inputs_dir=Path("inputs"),
                        outputs_dir=Path(tmpdir),
                        step_name="step5_production",
                    )

        self.checkpoint_manager.clear.assert_not_called()

    def test_simulation_initialization_failure_does_not_create_output_directory(self) -> None:
        self.simulation_factory.create.side_effect = RuntimeError("CUDA initialization failed")

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "charmm_gui_md.shared.workflows.stage_runner.OpenMMNativeFiles.from_root",
                return_value=self.files,
            ),
            mock.patch(
                "charmm_gui_md.shared.workflows.stage_runner.OpenMMStageProtocol.from_file",
                return_value=self.protocol,
            ),
        ):
            outputs_dir = Path(tmpdir) / "new_replica"
            with self.assertRaisesRegex(RuntimeError, "CUDA initialization failed"):
                self._runner().run(
                    inputs_dir=Path("inputs"),
                    outputs_dir=outputs_dir,
                    step_name="step5_production",
                )
            self.assertFalse(outputs_dir.exists())

        self.checkpoint_manager.restore.assert_not_called()
        self.reporter_installer.install.assert_not_called()

    def test_simulation_initialization_failure_preserves_existing_output(self) -> None:
        self.simulation_factory.create.side_effect = RuntimeError("CUDA initialization failed")

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "charmm_gui_md.shared.workflows.stage_runner.OpenMMNativeFiles.from_root",
                return_value=self.files,
            ),
            mock.patch(
                "charmm_gui_md.shared.workflows.stage_runner.OpenMMStageProtocol.from_file",
                return_value=self.protocol,
            ),
        ):
            outputs_dir = Path(tmpdir)
            stage_dir = outputs_dir / "step5_production"
            stage_dir.mkdir()
            checkpoint = stage_dir / "checkpoint.chk"
            checkpoint.write_bytes(b"existing checkpoint")
            with self.assertRaisesRegex(RuntimeError, "CUDA initialization failed"):
                self._runner().run(
                    inputs_dir=Path("inputs"),
                    outputs_dir=outputs_dir,
                    step_name="step5_production",
                )
            self.assertEqual(checkpoint.read_bytes(), b"existing checkpoint")

        self.checkpoint_manager.restore.assert_not_called()
        self.checkpoint_manager.clear.assert_not_called()


if __name__ == "__main__":
    unittest.main()
