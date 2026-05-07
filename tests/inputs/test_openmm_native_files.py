import os

os.environ["MEMBRANE_OPENMM_SKIP_BOOTSTRAP"] = "1"

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from protein_membrane_md.inputs.openmm_native_files import OpenmmNativeFiles


VALID_TOPPAR_FILES = (
    *(f"toppar_all36_lipid_{index:02d}.str" for index in range(22)),
    *(f"toppar_all36_prot_{index:02d}.str" for index in range(18)),
    "top_all36_prot.rtf",
    "par_all36m_prot.prm",
    "top_all36_na.rtf",
    "par_all36_na.prm",
    "top_all36_carb.rtf",
    "par_all36_carb.prm",
    "top_all36_lipid.rtf",
    "par_all36_lipid.prm",
    "top_all36_cgenff.rtf",
    "par_all36_cgenff.prm",
    "top_interface.rtf",
    "par_interface.prm",
    "toppar_water_ions.str",
    "toppar_dum_noble_gases.str",
    "toppar_ions_won.str",
    "cam.str",
)


class OpenmmNativeFilesValidationTests(unittest.TestCase):
    def test_from_root_accepts_required_lig_and_toppar_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(Path(tmpdir))

            files = OpenmmNativeFiles.from_root(root)

        self.assertEqual(files.inputs_dir, (root / "openmm").resolve())

    def test_from_root_requires_lig_parameter_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(
                Path(tmpdir),
                missing_lig_files=("lig.prm",),
            )

            with self.assertRaisesRegex(ValueError, r"lig\.prm"):
                OpenmmNativeFiles.from_root(root)

    def test_from_root_rejects_wrong_toppar_file_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(
                Path(tmpdir),
                toppar_files=VALID_TOPPAR_FILES[:-1],
            )

            with self.assertRaisesRegex(ValueError, r"Expected 56 files"):
                OpenmmNativeFiles.from_root(root)

    def test_from_root_rejects_toppar_without_enough_lipid_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(
                Path(tmpdir),
                toppar_files=(
                    *(f"toppar_all36_sterol_{index:02d}.str" for index in range(40)),
                    *(f"top_all36_extra_{index:02d}.rtf" for index in range(16)),
                ),
            )

            with self.assertRaisesRegex(ValueError, r"lipid"):
                OpenmmNativeFiles.from_root(root)

    def test_params_file_uses_ordered_references_from_toppar_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_openmm_native_fixture(
                Path(tmpdir),
                toppar_stream=(
                    "../toppar/top_all36_prot.rtf\n"
                    "stream ../toppar/toppar_all36_lipid_00.str ! comment\n"
                    "open read card unit 10 name ../lig/lig.rtf\n"
                    "../toppar/top_all36_prot.rtf\n"
                ),
            )
            files = OpenmmNativeFiles.from_root(root)
            params = object()

            with mock.patch(
                "protein_membrane_md.inputs.openmm_native_files.CharmmParameterSet",
                return_value=params,
            ) as parameter_set:
                self.assertIs(files.params_file, params)

        expected_paths = (
            root / "toppar" / "top_all36_prot.rtf",
            root / "toppar" / "toppar_all36_lipid_00.str",
            root / "lig" / "lig.rtf",
        )
        self.assertEqual(
            parameter_set.call_args.args,
            tuple(str(path.resolve()) for path in expected_paths),
        )

    def test_runtime_parameter_path_resolution_is_a_single_helper(self) -> None:
        self.assertTrue(
            hasattr(OpenmmNativeFiles, "_parameter_paths_from_toppar_stream")
        )
        self.assertFalse(hasattr(OpenmmNativeFiles, "_parameter_references"))
        self.assertFalse(hasattr(OpenmmNativeFiles, "_resolve_reference"))


def _write_openmm_native_fixture(
    root: Path,
    *,
    missing_lig_files: tuple[str, ...] = (),
    toppar_files: tuple[str, ...] = VALID_TOPPAR_FILES,
    toppar_stream: str = "",
) -> Path:
    openmm_dir = root / "openmm"
    lig_dir = root / "lig"
    toppar_dir = root / "toppar"

    openmm_dir.mkdir()
    lig_dir.mkdir()
    toppar_dir.mkdir()

    for name in OpenmmNativeFiles.REQUIRED_FILES:
        if name == "sysinfo.dat":
            contents = '{"dimensions": [75.0, 75.0, 95.0]}'
        elif name == "toppar.str":
            contents = toppar_stream
        else:
            contents = ""
        (openmm_dir / name).write_text(contents)

    for name in ("lig.prm", "lig.rtf"):
        if name not in missing_lig_files:
            (lig_dir / name).write_text("")

    for name in toppar_files:
        (toppar_dir / name).write_text("")

    return root


if __name__ == "__main__":
    unittest.main()
