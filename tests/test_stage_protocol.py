import os

os.environ["MEMBRANE_OPENMM_SKIP_BOOTSTRAP"] = "1"

import unittest
from pathlib import Path

from protein_membrane_md.protocols import OpenMMStageProtocol


class OpenMMStageProtocolTests(unittest.TestCase):
    def test_from_file_reads_charmm_gui_langevin_stage(self) -> None:
        protocol = OpenMMStageProtocol.from_file(
            step_name="step6.1_equilibration",
            protocol_path=Path("data/inputs/charmmgui/step6.1_equilibration.inp"),
        )

        self.assertEqual(protocol.dynamics_steps, 125000)
        self.assertEqual(protocol.minimization_steps, 3000)
        self.assertTrue(protocol.generate_velocities)
        self.assertEqual(protocol.timestep_ps, 0.001)
        self.assertEqual(protocol.trajectory_report_interval_steps, 5000)
        self.assertEqual(protocol.switch_distance_nm, 1.0)
        self.assertEqual(protocol.cutoff_distance_nm, 1.2)
        self.assertFalse(protocol.pressure_coupling)

    def test_from_file_reads_charmm_gui_membrane_pressure_stage(self) -> None:
        protocol = OpenMMStageProtocol.from_file(
            step_name="step6.3_equilibration",
            protocol_path=Path("data/inputs/charmmgui/step6.3_equilibration.inp"),
        )

        self.assertEqual(protocol.dynamics_steps, 125000)
        self.assertFalse(protocol.generate_velocities)
        self.assertTrue(protocol.pressure_coupling)
        self.assertEqual(protocol.barostat_kind, "membrane")
        self.assertEqual(protocol.pressure_bar, 1.0)
        self.assertEqual(protocol.barostat_interval_steps, 15)


if __name__ == "__main__":
    unittest.main()
