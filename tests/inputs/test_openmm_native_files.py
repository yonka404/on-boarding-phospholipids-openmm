import os

os.environ["CHARMM_GUI_MD_OPENMM_SKIP_BOOTSTRAP"] = "1"

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from charmm_gui_md.membrane.profile import MEMBRANE_PROFILE
from charmm_gui_md.shared.inputs.openmm_native_files import OpenMMNativeFiles
from charmm_gui_md.shared.profile import SystemProfile
from charmm_gui_md.solution.profile import SOLUTION_PROFILE


class OpenMMNativeFilesValidationTests(unittest.TestCase):
    def test_membrane_profile_accepts_only_referenced_parameter_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(
                Path(tmpdir),
                profile=MEMBRANE_PROFILE,
                toppar_stream=(
                    "../toppar/top_all36_prot.rtf\n"
                    "../lig/lig.rtf\n"
                ),
                referenced_files=(
                    "toppar/top_all36_prot.rtf",
                    "lig/lig.rtf",
                ),
            )

            files = OpenMMNativeFiles.from_root(root, profile=MEMBRANE_PROFILE)

        self.assertEqual(files.inputs_dir, (root / "openmm").resolve())
        self.assertEqual(files.profile, MEMBRANE_PROFILE)

    def test_solution_profile_accepts_bundle_without_lig_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(
                Path(tmpdir),
                profile=SOLUTION_PROFILE,
                toppar_stream="../toppar/top_all36_prot.rtf\n",
                referenced_files=("toppar/top_all36_prot.rtf",),
            )

            files = OpenMMNativeFiles.from_root(root, profile=SOLUTION_PROFILE)

        self.assertEqual(files.initial_coordinates_path.name, "step3_input.pdb")

    def test_from_root_rejects_missing_referenced_parameter_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(
                Path(tmpdir),
                profile=SOLUTION_PROFILE,
                toppar_stream="../toppar/missing.str\n",
            )

            with self.assertRaisesRegex(ValueError, r"missing\.str"):
                OpenMMNativeFiles.from_root(root, profile=SOLUTION_PROFILE)

    def test_from_root_requires_profile_specific_initial_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(
                Path(tmpdir),
                profile=SOLUTION_PROFILE,
                missing_openmm_files=("step3_input.crd",),
            )

            with self.assertRaisesRegex(ValueError, r"step3_input\.crd"):
                OpenMMNativeFiles.from_root(root, profile=SOLUTION_PROFILE)

    def test_from_root_requires_profile_specific_stage_protocols(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(
                Path(tmpdir),
                profile=SOLUTION_PROFILE,
                missing_openmm_files=("step4_equilibration.inp",),
            )

            with self.assertRaisesRegex(ValueError, r"step4_equilibration\.inp"):
                OpenMMNativeFiles.from_root(root, profile=SOLUTION_PROFILE)

    def test_params_file_uses_ordered_references_from_toppar_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(
                Path(tmpdir),
                profile=MEMBRANE_PROFILE,
                toppar_stream=(
                    "../toppar/top_all36_prot.rtf\n"
                    "../toppar/toppar_water_ions.str ! comment\n"
                    "../lig/lig.rtf\n"
                    "../toppar/top_all36_prot.rtf\n"
                ),
                referenced_files=(
                    "toppar/top_all36_prot.rtf",
                    "toppar/toppar_water_ions.str",
                    "lig/lig.rtf",
                ),
            )
            files = OpenMMNativeFiles.from_root(root, profile=MEMBRANE_PROFILE)
            params = object()

            with mock.patch(
                "charmm_gui_md.shared.inputs.openmm_native_files.CharmmParameterSet",
                return_value=params,
            ) as parameter_set:
                self.assertIs(files.params_file, params)

        expected_paths = (
            root / "toppar" / "top_all36_prot.rtf",
            root / "toppar" / "toppar_water_ions.str",
            root / "lig" / "lig.rtf",
        )
        self.assertEqual(
            parameter_set.call_args.args,
            tuple(str(path.resolve()) for path in expected_paths),
        )

    def test_solution_native_file_properties_use_profile_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(Path(tmpdir), profile=SOLUTION_PROFILE)
            files = OpenMMNativeFiles.from_root(root, profile=SOLUTION_PROFILE)
            crd_file = mock.Mock(positions=object())

            with mock.patch(
                "charmm_gui_md.shared.inputs.openmm_native_files.CharmmCrdFile",
                return_value=crd_file,
            ) as crd_file_ctor:
                self.assertIs(files.restraint_reference_positions, crd_file.positions)

        crd_file_ctor.assert_called_once_with(str(root / "openmm" / "step3_input.crd"))

    def test_active_protein_restraints_require_protein_position_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(
                Path(tmpdir),
                profile=SOLUTION_PROFILE,
                protocol_overrides={
                    "step4_equilibration": "rest = yes\nfc_bb = 400.0\n",
                },
            )

            with self.assertRaisesRegex(ValueError, r"prot_pos\.txt"):
                OpenMMNativeFiles.from_root(root, profile=SOLUTION_PROFILE)

    def test_active_lipid_restraints_require_lipid_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(
                Path(tmpdir),
                profile=MEMBRANE_PROFILE,
                protocol_overrides={
                    "step6.1_equilibration": (
                        "rest = yes\nfc_lpos = 100.0\nfc_ldih = 50.0\n"
                    ),
                },
            )

            with self.assertRaisesRegex(ValueError, r"(?s)lipid_pos\.txt.*dihe\.txt"):
                OpenMMNativeFiles.from_root(root, profile=MEMBRANE_PROFILE)

    def test_disabled_restraints_do_not_require_restraint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(Path(tmpdir), profile=SOLUTION_PROFILE)

            files = OpenMMNativeFiles.from_root(root, profile=SOLUTION_PROFILE)

        self.assertEqual(files.profile, SOLUTION_PROFILE)

    def test_from_root_rejects_raw_charmm_toppar_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(
                Path(tmpdir),
                profile=SOLUTION_PROFILE,
                toppar_stream="stream ../toppar/toppar_water_ions.str\n",
                referenced_files=("toppar/toppar_water_ions.str",),
            )

            with self.assertRaisesRegex(ValueError, r"Unsupported.*toppar\.str"):
                OpenMMNativeFiles.from_root(root, profile=SOLUTION_PROFILE)

    def test_from_root_rejects_non_json_sysinfo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(Path(tmpdir), profile=SOLUTION_PROFILE)
            (root / "openmm" / "sysinfo.dat").write_text(
                "BOXLX = 75.0\nBOXLY = 75.0\nBOXLZ = 95.0\n"
            )

            with self.assertRaisesRegex(ValueError, r"Invalid JSON"):
                OpenMMNativeFiles.from_root(root, profile=SOLUTION_PROFILE)


def _write_openmm_native_fixture(
    root: Path,
    *,
    profile: SystemProfile,
    toppar_stream: str = "",
    referenced_files: tuple[str, ...] = (),
    missing_openmm_files: tuple[str, ...] = (),
    protocol_overrides: dict[str, str] | None = None,
) -> Path:
    openmm_dir = root / "openmm"
    openmm_dir.mkdir()

    initial_files = tuple(
        f"{profile.initial_input_prefix}.{suffix}" for suffix in ("psf", "pdb", "crd")
    )
    stage_files = tuple(f"{name}.inp" for name in profile.protocol_schedule.stage_names)
    required_files = (*initial_files, "sysinfo.dat", "toppar.str", *stage_files)

    for name in required_files:
        if name in missing_openmm_files:
            continue
        if name == "sysinfo.dat":
            contents = '{"dimensions": [75.0, 75.0, 95.0]}'
        elif name == "toppar.str":
            contents = toppar_stream
        elif name.endswith(".inp"):
            step_name = name.removesuffix(".inp")
            contents = _protocol_text(
                (protocol_overrides or {}).get(step_name, ""),
            )
        else:
            contents = ""
        (openmm_dir / name).write_text(contents)

    for relative_name in referenced_files:
        path = root / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    return root


def _protocol_text(extra: str = "") -> str:
    return (
        "nstep = 10\n"
        "dt = 0.001\n"
        "nstout = 10\n"
        "nstdcd = 10\n"
        "temp = 303.15\n"
        "fric_coeff = 1.0\n"
        "r_on = 1.0\n"
        "r_off = 1.2\n"
        "ewald_tol = 0.0005\n"
        "rest = no\n"
        f"{extra}"
    )


if __name__ == "__main__":
    unittest.main()
