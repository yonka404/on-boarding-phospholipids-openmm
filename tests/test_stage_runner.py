import os

os.environ["MEMBRANE_OPENMM_SKIP_BOOTSTRAP"] = "1"

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from protein_membrane_md.workflows.stage_runner import StageRunner


class StageRunnerTests(unittest.TestCase):
    def test_run_writes_stages_under_given_output_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs_dir = root / "data" / "inputs" / "charmmgui"
            outputs_dir = root / "data" / "outputs" / "charmmgui"
            inputs_dir.mkdir(parents=True)

            files = SimpleNamespace(inputs_dir=inputs_dir)
            protocol = SimpleNamespace(
                step_name="step6.1_equilibration",
                has_minimization=False,
                dynamics_steps=10,
                timestep_ps=0.001,
            )
            simulation = mock.Mock()
            output_writer = mock.Mock(
                write=mock.Mock(
                    return_value=outputs_dir
                    / "step6.1_equilibration"
                    / "final_coordinates.pdb"
                )
            )
            restart_resolver = mock.Mock(
                resolve=mock.Mock(return_value=mock.sentinel.restart_source)
            )
            reporter_installer = mock.Mock()

            with (
                mock.patch(
                    "protein_membrane_md.workflows.stage_runner.CharmmGuiFiles"
                ) as charmm_gui_files,
                mock.patch(
                    "protein_membrane_md.workflows.stage_runner.OpenMMStageProtocol"
                ) as openmm_stage_protocol,
            ):
                charmm_gui_files.from_root.return_value = files
                openmm_stage_protocol.from_file.return_value = protocol

                result = StageRunner(
                    restart_resolver=restart_resolver,
                    simulation_factory=mock.Mock(
                        create=mock.Mock(return_value=simulation)
                    ),
                    simulation_initializer=mock.Mock(),
                    reporter_installer=reporter_installer,
                    output_writer=output_writer,
                ).run(
                    inputs_dir=inputs_dir,
                    outputs_dir=outputs_dir,
                    step_name="step6.1_equilibration",
                )

            expected_stage_dir = outputs_dir / "step6.1_equilibration"

            openmm_stage_protocol.from_file.assert_called_once_with(
                step_name="step6.1_equilibration",
                protocol_path=inputs_dir / "step6.1_equilibration.inp",
            )
            restart_resolver.resolve.assert_called_once_with(
                inputs_dir=inputs_dir,
                outputs_dir=outputs_dir,
                step_name="step6.1_equilibration",
            )
            self.assertEqual(
                reporter_installer.install.call_args.args[1].output_dir,
                expected_stage_dir,
            )
            self.assertTrue(expected_stage_dir.is_dir())
            self.assertEqual(
                result,
                expected_stage_dir / "final_coordinates.pdb",
            )


if __name__ == "__main__":
    unittest.main()
