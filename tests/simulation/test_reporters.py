import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from protein_membrane_md.artifacts import StageArtifacts
from protein_membrane_md.simulation.reporters import StageReporterInstaller


class StageReporterInstallerTests(unittest.TestCase):
    def test_dcd_reporter_uses_periodic_imaging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = StageArtifacts.for_stage(Path(tmpdir), "step7_production")
            simulation = SimpleNamespace(reporters=[])
            protocol = SimpleNamespace(
                step_name="step7_production",
                state_report_interval_steps=10,
                trajectory_report_interval_steps=25,
            )

            with (
                mock.patch(
                    "protein_membrane_md.simulation.reporters.StateDataReporter",
                    return_value="state-reporter",
                ),
                mock.patch(
                    "protein_membrane_md.simulation.reporters.DCDReporter",
                    return_value="dcd-reporter",
                ) as dcd_reporter,
            ):
                StageReporterInstaller().install(simulation, artifacts, protocol)

        dcd_reporter.assert_called_once_with(
            str(artifacts.trajectory_path),
            25,
            enforcePeriodicBox=True,
        )
        self.assertEqual(simulation.reporters, ["state-reporter", "dcd-reporter"])


if __name__ == "__main__":
    unittest.main()
