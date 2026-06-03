import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from charmm_gui_md.shared.artifacts import StageArtifacts
from charmm_gui_md.shared.simulation.outputs import StageOutputWriter


class StageOutputWriterTests(unittest.TestCase):
    def test_final_coordinates_are_written_with_periodic_imaging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = StageArtifacts.for_stage(Path(tmpdir), "step7_production")
            artifacts.create_directory()
            simulation = mock.Mock()
            state = mock.Mock()
            simulation.context.getState.return_value = state

            with mock.patch(
                "charmm_gui_md.shared.simulation.outputs.PDBFile.writeFile",
            ) as write_file:
                StageOutputWriter().write(
                    simulation,
                    artifacts,
                    SimpleNamespace(step_name="step7_production"),
                )

        simulation.saveState.assert_called_once_with(str(artifacts.final_state_path))
        simulation.context.getState.assert_called_once_with(
            getPositions=True,
            enforcePeriodicBox=True,
        )
        simulation.topology.setPeriodicBoxVectors.assert_called_once_with(
            state.getPeriodicBoxVectors.return_value,
        )
        write_file.assert_called_once()
        self.assertIs(write_file.call_args.args[0], simulation.topology)
        self.assertIs(write_file.call_args.args[1], state.getPositions.return_value)


if __name__ == "__main__":
    unittest.main()
