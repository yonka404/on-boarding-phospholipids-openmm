import json
from pathlib import Path
from typing import ClassVar

from openmm.app import CharmmParameterSet, CharmmPsfFile, PDBFile
from pydantic import BaseModel, field_validator


class OpenMMNativeFiles(BaseModel):
    """Validated CHARMM-GUI OpenMM-native files needed for the OpenMM setup."""

    model_config = {"frozen": True}

    OPENMM_SUBDIRECTORY: ClassVar[str] = "openmm"
    REQUIRED_PARENT_DIRECTORIES: ClassVar[tuple[str, ...]] = (
        "lig",
        "toppar",
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
    def from_root(cls, inputs_dir: Path) -> "OpenMMNativeFiles":
        """Construct from an OpenMM-native root or its nested openmm directory."""
        root = Path(inputs_dir).expanduser()
        openmm_dir = root / cls.OPENMM_SUBDIRECTORY
        return cls(inputs_dir=openmm_dir if openmm_dir.is_dir() else root)

    @field_validator("inputs_dir", mode="after")
    @classmethod
    def _validate_inputs_root(cls, v: Path) -> Path:
        root = v.expanduser().resolve()

        if not root.is_dir():
            raise ValueError(f"inputs_dir is not a directory: {root}")

        missing_dirs = [
            root.parent / name
            for name in cls.REQUIRED_PARENT_DIRECTORIES
            if not (root.parent / name).is_dir()
        ]
        missing_files = [
            root / name for name in cls.REQUIRED_FILES if not (root / name).is_file()
        ]

        if missing_dirs or missing_files:
            missing = [*missing_dirs, *missing_files]
            joined = "\n".join(f"  - {p}" for p in missing)
            raise ValueError(f"Missing required OpenMM-native files:\n{joined}")

        parameter_paths = cls._parameter_file_paths(root)
        if not parameter_paths:
            raise ValueError(
                f"No topology/parameter files referenced by {root / 'toppar.str'}"
            )

        missing_parameter_files = [path for path in parameter_paths if not path.is_file()]
        if missing_parameter_files:
            joined = "\n".join(f"  - {p}" for p in missing_parameter_files)
            raise ValueError(
                f"Missing topology/parameter files referenced by toppar.str:\n{joined}"
            )

        cls._sysinfo_box_lengths(root / "sysinfo.dat")
        return root

    @classmethod
    def _parameter_file_paths(cls, root: Path) -> tuple[Path, ...]:
        paths: list[Path] = []
        seen: set[Path] = set()

        for reference in cls._parameter_references(root / "toppar.str"):
            path = cls._resolve_reference(root, reference)
            if path not in seen:
                paths.append(path)
                seen.add(path)

        return tuple(paths)

    @staticmethod
    def _parameter_references(toppar_file: Path) -> tuple[str, ...]:
        references: list[str] = []

        for raw_line in toppar_file.read_text().splitlines():
            line = raw_line.split("!", maxsplit=1)[0].split("#", maxsplit=1)[0].strip()
            if not line:
                continue

            words = line.split()
            command = words[0].lower()
            if command == "stream" and len(words) > 1:
                references.append(words[1])
            elif command == "open":
                lowered = [word.lower() for word in words]
                if "name" in lowered:
                    name_index = lowered.index("name") + 1
                    if name_index < len(words):
                        references.append(words[name_index])
            else:
                references.append(words[0])

        return tuple(references)

    @staticmethod
    def _resolve_reference(root: Path, reference: str) -> Path:
        ref_path = Path(reference)
        candidates = [ref_path] if ref_path.is_absolute() else [root / ref_path]

        if not ref_path.is_absolute():
            stripped_parts = tuple(
                part for part in ref_path.parts if part not in {".", ".."}
            )
            if stripped_parts:
                candidates.append(root.joinpath(*stripped_parts))

        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved.is_file():
                return resolved

        return candidates[0].expanduser().resolve()

    @property
    def psf_file(self) -> CharmmPsfFile:
        return CharmmPsfFile(str(self.inputs_dir / "step5_input.psf"))

    @property
    def pdb_file(self) -> PDBFile:
        return PDBFile(str(self.inputs_dir / "step5_input.pdb"))

    @property
    def initial_coordinates_path(self) -> Path:
        return self.inputs_dir / "step5_input.pdb"

    @property
    def initial_coordinates_description(self) -> str:
        return "initial CHARMM-GUI OpenMM coordinates"

    @property
    def params_file(self) -> CharmmParameterSet:
        return CharmmParameterSet(
            *(str(path) for path in self._parameter_file_paths(self.inputs_dir))
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
            return OpenMMNativeFiles._legacy_sysinfo_box_lengths(sysinfo_file, text)

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


OpenmmNativeFiles = OpenMMNativeFiles
