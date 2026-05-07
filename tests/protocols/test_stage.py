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

    def test_from_file_reads_openmm_native_restraint_fields(self) -> None:
        protocol = OpenMMStageProtocol.from_file(
            step_name="step6.1_equilibration",
            protocol_path=Path(
                "data/inputs/openmm_native/openmm/step6.1_equilibration.inp"
            ),
        )

        self.assertEqual(protocol.coulomb_method_name, "PME")
        self.assertEqual(protocol.vdw_method_name, "Force-switch")
        self.assertTrue(protocol.restraints_enabled)
        self.assertEqual(protocol.protein_backbone_restraint_kj_mol_nm2, 4000.0)
        self.assertEqual(protocol.protein_side_chain_restraint_kj_mol_nm2, 2000.0)
        self.assertEqual(protocol.lipid_position_restraint_kj_mol_nm2, 1000.0)
        self.assertEqual(protocol.lipid_dihedral_restraint_kj_mol_rad2, 1000.0)
        self.assertEqual(protocol.carbohydrate_dihedral_restraint_kj_mol_rad2, 1000.0)

    def test_from_file_defaults_missing_openmm_native_restraint_constants(self) -> None:
        protocol = OpenMMStageProtocol.from_file(
            step_name="step6.6_equilibration",
            protocol_path=Path(
                "data/inputs/openmm_native/openmm/step6.6_equilibration.inp"
            ),
        )

        self.assertTrue(protocol.restraints_enabled)
        self.assertEqual(protocol.protein_backbone_restraint_kj_mol_nm2, 50.0)
        self.assertEqual(protocol.protein_side_chain_restraint_kj_mol_nm2, 0.0)
        self.assertEqual(protocol.lipid_position_restraint_kj_mol_nm2, 0.0)
        self.assertEqual(protocol.lipid_dihedral_restraint_kj_mol_rad2, 0.0)
        self.assertEqual(protocol.carbohydrate_dihedral_restraint_kj_mol_rad2, 0.0)

    def test_from_file_reads_openmm_native_restraints_disabled(self) -> None:
        protocol = OpenMMStageProtocol.from_file(
            step_name="step7_production",
            protocol_path=Path("data/inputs/openmm_native/openmm/step7_production.inp"),
        )

        self.assertFalse(protocol.restraints_enabled)
        self.assertEqual(protocol.protein_backbone_restraint_kj_mol_nm2, 0.0)
        self.assertEqual(protocol.protein_side_chain_restraint_kj_mol_nm2, 0.0)
        self.assertEqual(protocol.lipid_position_restraint_kj_mol_nm2, 0.0)
        self.assertEqual(protocol.lipid_dihedral_restraint_kj_mol_rad2, 0.0)
        self.assertEqual(protocol.carbohydrate_dihedral_restraint_kj_mol_rad2, 0.0)

    def test_from_file_models_all_current_openmm_native_assignment_keys(self) -> None:
        field_by_key = {
            "mini_nstep": ("minimization_steps", int),
            "mini_tol": ("minimization_tolerance_kj_mol_nm", float),
            "gen_vel": ("generate_velocities", _yes_no),
            "gen_temp": ("velocity_temperature_kelvin", float),
            "nstep": ("dynamics_steps", int),
            "dt": ("timestep_ps", float),
            "nstout": ("state_report_interval_steps", int),
            "nstdcd": ("trajectory_report_interval_steps", int),
            "coulomb": ("coulomb_method_name", str),
            "ewald_tol": ("ewald_tolerance", float),
            "vdw": ("vdw_method_name", str),
            "r_on": ("switch_distance_nm", float),
            "r_off": ("cutoff_distance_nm", float),
            "temp": ("temperature_kelvin", float),
            "fric_coeff": ("friction_per_ps", float),
            "pcouple": ("pressure_coupling", _yes_no),
            "p_ref": ("pressure_bar", float),
            "p_type": ("barostat_kind", str),
            "p_xymode": ("membrane_xy_mode_name", str),
            "p_zmode": ("membrane_z_mode_name", str),
            "p_tens": ("surface_tension_dyne_per_cm", float),
            "p_freq": ("barostat_interval_steps", int),
            "cons": ("constraints_name", str),
            "rest": ("restraints_enabled", _yes_no),
            "fc_bb": ("protein_backbone_restraint_kj_mol_nm2", float),
            "fc_sc": ("protein_side_chain_restraint_kj_mol_nm2", float),
            "fc_lpos": ("lipid_position_restraint_kj_mol_nm2", float),
            "fc_ldih": ("lipid_dihedral_restraint_kj_mol_rad2", float),
            "fc_cdih": ("carbohydrate_dihedral_restraint_kj_mol_rad2", float),
        }

        for protocol_path in sorted(Path("data/inputs/openmm_native/openmm").glob("step*.inp")):
            with self.subTest(protocol_path=protocol_path):
                assignments = _assignment_values(protocol_path)
                protocol = OpenMMStageProtocol.from_file(
                    step_name=protocol_path.stem,
                    protocol_path=protocol_path,
                )

                for key, (field_name, converter) in field_by_key.items():
                    if key not in assignments:
                        continue

                    self.assertEqual(
                        getattr(protocol, field_name),
                        converter(assignments[key]),
                        f"{protocol_path.name} {key}",
                    )


def _assignment_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", maxsplit=1)[0].split("!", maxsplit=1)[0].strip()
        if not line or "=" not in line:
            continue

        key, value = (part.strip() for part in line.split("=", maxsplit=1))
        values[key.lower()] = value

    return values


def _yes_no(value: str) -> bool:
    return value.strip().lower() == "yes"


if __name__ == "__main__":
    unittest.main()
