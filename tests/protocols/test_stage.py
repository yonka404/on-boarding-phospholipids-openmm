import os

os.environ["CHARMM_GUI_MD_OPENMM_SKIP_BOOTSTRAP"] = "1"

import tempfile
import unittest
from pathlib import Path

from charmm_gui_md.shared.protocols import OpenMMStageProtocol


MEMBRANE_OPENMM_DIR = Path(
    "data/inputs/openmm_native/membrane/ligand_membrane/openmm"
)
SOLUTION_OPENMM_DIR = Path("data/inputs/openmm_native/solution/abeta_40/openmm")


class OpenMMStageProtocolTests(unittest.TestCase):
    def test_from_file_rejects_raw_charmm_protocol_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol_path = Path(tmpdir) / "stage.inp"
            protocol_path.write_text(
                "dyna start nstep 125000 timestp 0.001 finalt 303.15\n"
            )

            with self.assertRaisesRegex(ValueError, r"Missing integer protocol field"):
                OpenMMStageProtocol.from_file(
                    step_name="stage",
                    protocol_path=protocol_path,
                )

    def test_from_file_reads_openmm_native_restraint_fields(self) -> None:
        protocol = OpenMMStageProtocol.from_file(
            step_name="step6.1_equilibration",
            protocol_path=MEMBRANE_OPENMM_DIR / "step6.1_equilibration.inp",
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
            protocol_path=MEMBRANE_OPENMM_DIR / "step6.6_equilibration.inp",
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
            protocol_path=MEMBRANE_OPENMM_DIR / "step7_production.inp",
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

        protocol_paths = (
            *sorted(MEMBRANE_OPENMM_DIR.glob("step*.inp")),
            *sorted(SOLUTION_OPENMM_DIR.glob("step*.inp")),
        )
        for protocol_path in protocol_paths:
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

    def test_from_file_reads_solution_equilibration_restraints(self) -> None:
        protocol = OpenMMStageProtocol.from_file(
            step_name="step4_equilibration",
            protocol_path=SOLUTION_OPENMM_DIR / "step4_equilibration.inp",
        )

        self.assertTrue(protocol.restraints_enabled)
        self.assertEqual(protocol.protein_backbone_restraint_kj_mol_nm2, 400.0)
        self.assertEqual(protocol.protein_side_chain_restraint_kj_mol_nm2, 40.0)
        self.assertFalse(protocol.pressure_coupling)

    def test_from_file_reads_solution_isotropic_pressure_stage(self) -> None:
        protocol = OpenMMStageProtocol.from_file(
            step_name="step5_production",
            protocol_path=SOLUTION_OPENMM_DIR / "step5_production.inp",
        )

        self.assertFalse(protocol.restraints_enabled)
        self.assertTrue(protocol.pressure_coupling)
        self.assertEqual(protocol.barostat_kind, "isotropic")
        self.assertEqual(protocol.pressure_bar, 1.0)


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
