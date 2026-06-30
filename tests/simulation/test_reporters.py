import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from charmm_gui_md.shared.artifacts import StageArtifacts
from charmm_gui_md.shared.simulation.reporters import StageReporterInstaller


class StageReporterInstallerTests(unittest.TestCase):
    def test_reporters_append_outputs_and_write_periodic_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = StageArtifacts.for_stage(Path(tmpdir), "step7_production")
            artifacts.create_directory()
            artifacts.state_data_path.write_text("existing data")
            artifacts.trajectory_path.write_bytes(b"existing trajectory")
            simulation = SimpleNamespace(reporters=[])
            protocol = SimpleNamespace(
                step_name="step7_production",
                state_report_interval_steps=10,
                trajectory_report_interval_steps=25,
            )

            with (
                mock.patch(
                    "charmm_gui_md.shared.simulation.reporters.StateDataReporter",
                    return_value="state-reporter",
                ) as state_reporter,
                mock.patch(
                    "charmm_gui_md.shared.simulation.reporters.DCDReporter",
                    return_value="dcd-reporter",
                ) as dcd_reporter,
                mock.patch(
                    "charmm_gui_md.shared.simulation.reporters.CheckpointReporter",
                    return_value="checkpoint-reporter",
                ) as checkpoint_reporter,
            ):
                StageReporterInstaller().install(
                    simulation,
                    artifacts,
                    protocol,
                    append=True,
                )

        self.assertTrue(state_reporter.call_args.kwargs["append"])
        dcd_reporter.assert_called_once_with(
            str(artifacts.trajectory_path),
            25,
            append=True,
            enforcePeriodicBox=True,
        )
        checkpoint_reporter.assert_called_once_with(
            str(artifacts.checkpoint_path),
            25,
        )
        self.assertEqual(
            simulation.reporters,
            ["state-reporter", "dcd-reporter", "checkpoint-reporter"],
        )

    def test_append_requires_both_existing_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = StageArtifacts.for_stage(Path(tmpdir), "step7_production")
            artifacts.create_directory()
            simulation = SimpleNamespace(reporters=[])
            protocol = SimpleNamespace(
                step_name="step7_production",
                state_report_interval_steps=10,
                trajectory_report_interval_steps=25,
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                r"Cannot resume.*state_data\.csv.*trajectory\.dcd",
            ):
                StageReporterInstaller().install(
                    simulation,
                    artifacts,
                    protocol,
                    append=True,
                )

            self.assertEqual(simulation.reporters, [])
            self.assertFalse(artifacts.state_data_path.exists())
            self.assertFalse(artifacts.trajectory_path.exists())


if __name__ == "__main__":
    unittest.main()
