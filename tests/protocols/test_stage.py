import tempfile
import unittest
from pathlib import Path

from charmm_gui_md.shared.protocols import OpenMMStageProtocol

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "openmm_native"
MEMBRANE_OPENMM_DIR = FIXTURES / "membrane"
SOLUTION_OPENMM_DIR = FIXTURES / "solution"

_REQUIRED_VALUES = {
    "nstep": "10",
    "dt": "0.001",
    "nstout": "2",
    "nstdcd": "4",
    "temp": "303.15",
    "fric_coeff": "1.0",
    "r_on": "1.0",
    "r_off": "1.2",
    "ewald_tol": "0.0005",
}


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

    def test_from_file_models_all_dedicated_fixture_assignment_keys(self) -> None:
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
            MEMBRANE_OPENMM_DIR / "step6.1_equilibration.inp",
            MEMBRANE_OPENMM_DIR / "step6.6_equilibration.inp",
            MEMBRANE_OPENMM_DIR / "step7_production.inp",
            SOLUTION_OPENMM_DIR / "step4_equilibration.inp",
            SOLUTION_OPENMM_DIR / "step5_production.inp",
        )
        seen_keys: set[str] = set()
        for protocol_path in protocol_paths:
            with self.subTest(protocol_path=protocol_path):
                assignments = _assignment_values(protocol_path)
                protocol = OpenMMStageProtocol.from_file(
                    step_name=protocol_path.stem,
                    protocol_path=protocol_path,
                )

                self.assertTrue(assignments, f"Empty protocol fixture: {protocol_path}")
                self.assertLessEqual(assignments.keys(), field_by_key.keys())
                seen_keys.update(assignments)
                for key, value in assignments.items():
                    field_name, converter = field_by_key[key]
                    self.assertEqual(
                        getattr(protocol, field_name),
                        converter(value),
                        f"{protocol_path.name} {key}",
                    )
        self.assertEqual(seen_keys, field_by_key.keys())

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

    def test_missing_protocol_fails_visibly(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            self.assertRaisesRegex(FileNotFoundError, "Stage protocol not found"),
        ):
            OpenMMStageProtocol.from_file("stage", Path(tmpdir) / "missing.inp")

    def test_each_required_field_must_be_present(self) -> None:
        for key in _REQUIRED_VALUES:
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, f"field: {key}$"),
            ):
                self._read_values(
                    {k: v for k, v in _REQUIRED_VALUES.items() if k != key}
                )

    def test_malformed_numeric_fields_are_rejected(self) -> None:
        for key in (*_REQUIRED_VALUES, "mini_nstep", "mini_tol", "gen_temp", "fc_bb"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._read_values({**_REQUIRED_VALUES, key: "invalid"})

    def test_fractional_integer_fields_are_rejected(self) -> None:
        for key in ("nstep", "nstout", "nstdcd", "mini_nstep"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._read_values({**_REQUIRED_VALUES, key: "1.5"})

    def test_unsupported_boolean_tokens_are_rejected(self) -> None:
        for key in ("gen_vel", "pcouple", "rest"):
            for token in ("", "1", "maybe"):
                with (
                    self.subTest(key=key, token=token),
                    self.assertRaisesRegex(
                        ValueError, f"Unsupported boolean token for {key}"
                    ),
                ):
                    self._read_values({**_REQUIRED_VALUES, key: token})

    def test_pressure_coupling_requires_pressure_and_interval(self) -> None:
        values = {**_REQUIRED_VALUES, "pcouple": "yes", "p_ref": "1.0", "p_freq": "25"}
        for key in ("p_ref", "p_freq"):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, f"field: {key}$"),
            ):
                self._read_values({k: v for k, v in values.items() if k != key})

    def test_malformed_pressure_fields_are_rejected_when_coupling_is_enabled(
        self,
    ) -> None:
        values = {**_REQUIRED_VALUES, "pcouple": "yes", "p_ref": "1.0", "p_freq": "25"}
        for key in ("p_ref", "p_freq", "p_tens"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._read_values({**values, key: "invalid"})

    def test_missing_optional_fields_use_defaults(self) -> None:
        protocol = self._read_values(_REQUIRED_VALUES)
        expected = {
            "minimization_steps": None,
            "minimization_tolerance_kj_mol_nm": None,
            "generate_velocities": False,
            "velocity_temperature_kelvin": None,
            "constraints_name": None,
            "pressure_coupling": False,
            "pressure_bar": None,
            "barostat_kind": None,
            "membrane_xy_mode_name": None,
            "membrane_z_mode_name": None,
            "surface_tension_dyne_per_cm": None,
            "barostat_interval_steps": None,
            "coulomb_method_name": "PME",
            "vdw_method_name": "Force-switch",
            "restraints_enabled": False,
            "protein_backbone_restraint_kj_mol_nm2": 0.0,
            "protein_side_chain_restraint_kj_mol_nm2": 0.0,
            "lipid_position_restraint_kj_mol_nm2": 0.0,
            "lipid_dihedral_restraint_kj_mol_rad2": 0.0,
            "carbohydrate_dihedral_restraint_kj_mol_rad2": 0.0,
        }
        for name, value in expected.items():
            with self.subTest(field=name):
                self.assertEqual(getattr(protocol, name), value)
        self.assertFalse(protocol.has_minimization)

    def test_minimization_needs_both_step_count_and_tolerance(self) -> None:
        for extra, expected in (
            ({"mini_nstep": "20"}, False),
            ({"mini_tol": "10.0"}, False),
            ({"mini_nstep": "20", "mini_tol": "10.0"}, True),
        ):
            with self.subTest(extra=extra):
                self.assertEqual(
                    self._read_values({**_REQUIRED_VALUES, **extra}).has_minimization,
                    expected,
                )

    def test_assignments_accept_case_whitespace_comments_and_boolean_aliases(
        self,
    ) -> None:
        values = {**_REQUIRED_VALUES, " NSTEP ": "12 # steps", "DT": "0.002 ! ps"}
        for token, expected in (
            ("YES", True),
            (" True ", True),
            ("No", False),
            (" false ", False),
        ):
            with self.subTest(token=token):
                protocol = self._read_values(
                    {**values, "gen_vel": token, "rest": token}
                )
                self.assertEqual(protocol.dynamics_steps, 12)
                self.assertEqual(protocol.timestep_ps, 0.002)
                self.assertEqual(protocol.generate_velocities, expected)
                self.assertEqual(protocol.restraints_enabled, expected)

    def _read_values(self, values: dict[str, str]) -> OpenMMStageProtocol:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stage.inp"
            path.write_text(
                "\n".join(f"{key} = {value}" for key, value in values.items())
            )
            return OpenMMStageProtocol.from_file("stage", path)


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
