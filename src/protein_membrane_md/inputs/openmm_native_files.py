import json
import re
from pathlib import Path
from typing import ClassVar

from openmm.app import CharmmCrdFile, CharmmParameterSet, CharmmPsfFile, PDBFile
from pydantic import BaseModel, field_validator


class OpenmmNativeFiles(BaseModel):
    """Validated CHARMM-GUI OpenMM-native files needed for the OpenMM setup."""

    model_config = {"frozen": True}

    OPENMM_SUBDIRECTORY: ClassVar[str] = "openmm"
    REQUIRED_INPUT_ROOT_SUBDIRECTORIES: ClassVar[tuple[str, ...]] = (
        "lig",
        "toppar",
    )
    LIG_REQUIRED_FILES: ClassVar[tuple[str, ...]] = (
        "lig.prm",
        "lig.rtf",
    )
    TOPPAR_EXPECTED_FILE_COUNT: ClassVar[int] = 56
    TOPPAR_PATTERN_REQUIREMENTS: ClassVar[tuple[tuple[str, int, str], ...]] = (
        (r"\.(?:rtf|prm|str)\Z", 56, "CHARMM topology/parameter/stream files"),
        (r"\Atoppar_all36_.*\.str\Z", 40, "toppar_all36 stream files"),
        (r"lipid", 20, "lipid-related files"),
    )
    REQUIRED_FILES: ClassVar[tuple[str, ...]] = (
        "step5_input.psf",
        "step5_input.pdb",
        "step5_input.crd",
        "sysinfo.dat",
        "toppar.str",
        "step6.1_equilibration.inp",
        "step6.2_equilibration.inp",
        "step6.3_equilibration.inp",
        "step6.4_equilibration.inp",
        "step6.5_equilibration.inp",
        "step6.6_equilibration.inp",
        "step7_production.inp",
    )

    inputs_dir: Path

    @classmethod
    def from_root(cls, inputs_dir: Path) -> "OpenmmNativeFiles":
        openmm_dir = inputs_dir / cls.OPENMM_SUBDIRECTORY
        return cls(inputs_dir=openmm_dir)

    @field_validator("inputs_dir", mode="after")
    @classmethod
    def _validate_inputs_root(cls, v: Path) -> Path:
        root = v.resolve()

        # validate the openmm subfolder
        if not root.is_dir():
            raise ValueError(f"inputs_dir is not a directory: {root}")

        missing_dirs = [
            root.parent / name
            for name in cls.REQUIRED_INPUT_ROOT_SUBDIRECTORIES
            if not (root.parent / name).is_dir()
        ]

        # TODO: double-check here if I validate ALL files from openmm_native (restraints)
        missing_files = [
            root / name for name in cls.REQUIRED_FILES if not (root / name).is_file()
        ]

        if missing_dirs or missing_files:
            missing = [*missing_dirs, *missing_files]
            joined = "\n".join(f"  - {p}" for p in missing)
            raise ValueError(f"Missing required OpenMM-native files:\n{joined}")

        cls._validate_lig_directory(root.parent / "lig")
        cls._validate_toppar_directory(root.parent / "toppar")
        cls._sysinfo_box_lengths(root / "sysinfo.dat")

        return root

    @classmethod
    def _validate_lig_directory(cls, lig_dir: Path) -> None:
        missing_files = [
            lig_dir / name
            for name in cls.LIG_REQUIRED_FILES
            if not (lig_dir / name).is_file()
        ]
        if missing_files:
            joined = "\n".join(f"  - {p}" for p in missing_files)
            raise ValueError(f"Missing required OpenMM-native ligand files:\n{joined}")

    @classmethod
    def _validate_toppar_directory(cls, toppar_dir: Path) -> None:
        files = tuple(sorted(path for path in toppar_dir.iterdir() if path.is_file()))
        file_count = len(files)
        if file_count != cls.TOPPAR_EXPECTED_FILE_COUNT:
            raise ValueError(
                f"Expected {cls.TOPPAR_EXPECTED_FILE_COUNT} files in "
                f"{toppar_dir}, found {file_count}"
            )

        filenames = tuple(path.name for path in files)
        pattern_failures: list[str] = []
        for pattern, minimum, description in cls.TOPPAR_PATTERN_REQUIREMENTS:
            matched = sum(1 for name in filenames if re.search(pattern, name))
            if matched < minimum:
                pattern_failures.append(
                    f"  - expected at least {minimum} {description} "
                    f"matching /{pattern}/, found {matched}"
                )

        if pattern_failures:
            joined = "\n".join(pattern_failures)
            raise ValueError(
                f"Unexpected OpenMM-native toppar layout in {toppar_dir}:\n{joined}"
            )

    @classmethod
    def _parameter_paths_from_toppar_stream(cls, root: Path) -> tuple[Path, ...]:
        paths: list[Path] = []
        seen: set[Path] = set()

        for raw_line in (root / "toppar.str").read_text().splitlines():
            line = raw_line.split("!", maxsplit=1)[0].split("#", maxsplit=1)[0].strip()
            if not line:
                continue

            words = line.split()
            command = words[0].lower()
            reference = words[0]
            if command == "stream" and len(words) > 1:
                reference = words[1]
            elif command == "open":
                reference = ""
                lowered = [word.lower() for word in words]
                if "name" in lowered:
                    name_index = lowered.index("name") + 1
                    if name_index < len(words):
                        reference = words[name_index]
            if not reference:
                continue

            ref_path = Path(reference)
            candidates = [ref_path] if ref_path.is_absolute() else [root / ref_path]
            if not ref_path.is_absolute():
                stripped_parts = tuple(
                    part for part in ref_path.parts if part not in {".", ".."}
                )
                if stripped_parts:
                    candidates.append(root.joinpath(*stripped_parts))

            path = candidates[0].expanduser().resolve()
            for candidate in candidates:
                resolved = candidate.expanduser().resolve()
                if resolved.is_file():
                    path = resolved
                    break

            if path not in seen:
                paths.append(path)
                seen.add(path)

        return tuple(paths)

    @property
    def psf_file(self) -> CharmmPsfFile:
        return CharmmPsfFile(str(self.inputs_dir / "step5_input.psf"))

    @property
    def pdb_file(self) -> PDBFile:
        return PDBFile(str(self.inputs_dir / "step5_input.pdb"))

    @property
    def crd_file(self) -> CharmmCrdFile:
        return CharmmCrdFile(str(self.inputs_dir / "step5_input.crd"))

    @property
    def restraint_reference_positions(self):
        return self.crd_file.positions

    @property
    def initial_coordinates_path(self) -> Path:
        return self.inputs_dir / "step5_input.pdb"

    @property
    def params_file(self) -> CharmmParameterSet:
        # TODO: understand _parameter_paths_from_toppar_stream
        return CharmmParameterSet(
            *(
                str(path)
                for path in self._parameter_paths_from_toppar_stream(self.inputs_dir)
            )
        )

    @property
    def box_lengths_angstrom(self) -> tuple[float, float, float]:
        return self._sysinfo_box_lengths(self.inputs_dir / "sysinfo.dat")

    @staticmethod
    def _sysinfo_box_lengths(sysinfo_file: Path) -> tuple[float, float, float]:
        text = sysinfo_file.read_text()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return OpenmmNativeFiles._legacy_sysinfo_box_lengths(sysinfo_file, text)

        dimensions = data.get("dimensions")
        if not isinstance(dimensions, (list, tuple)) or len(dimensions) < 3:
            raise ValueError(f"Missing dimensions[0:3] in {sysinfo_file}")

        try:
            return float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid box dimensions in {sysinfo_file}") from exc

    @staticmethod
    def _legacy_sysinfo_box_lengths(
        sysinfo_file: Path,
        text: str,
    ) -> tuple[float, float, float]:
        # TODO: understand this method
        keys = {"BOXLX": "A", "BOXLY": "B", "BOXLZ": "C"}
        values: dict[str, float] = {}

        for raw_line in text.splitlines():
            line = raw_line.split("!", maxsplit=1)[0].split("#", maxsplit=1)[0]
            parts = line.split("=", maxsplit=1)
            if len(parts) != 2:
                continue

            source_key = parts[0].strip().upper()
            target_key = keys.get(source_key)
            if target_key is not None:
                values[target_key] = float(parts[1])

        missing = [key for key in ("A", "B", "C") if key not in values]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing box dimensions ({joined}) in {sysinfo_file}")

        return values["A"], values["B"], values["C"]
