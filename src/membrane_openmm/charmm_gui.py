from pathlib import Path
from typing import ClassVar

from openmm.app import CharmmParameterSet, CharmmPsfFile, PDBFile
from pydantic import BaseModel, field_validator


class CharmmGuiFiles(BaseModel):
    """Validated CHARMM-GUI files needed for the OpenMM setup."""

    model_config = {"frozen": True}

    REQUIRED_DIRECTORIES: ClassVar[tuple[str, ...]] = (
        "lig",
        "toppar",
    )

    REQUIRED_FILES: ClassVar[tuple[str, ...]] = (
        "membrane_restraint.charmm_openmm.str",
        "step5_assembly.psf",
        "step5_assembly.pdb",
        "step6.1_equilibration.inp",
        "step6.2_equilibration.inp",
        "step6.3_equilibration.inp",
        "step6.4_equilibration.inp",
        "step6.5_equilibration.inp",
        "step6.6_equilibration.inp",
        "step7_production.inp",
        "toppar.str",
    )

    inputs_dir: Path

    @classmethod
    def from_root(cls, inputs_dir: Path) -> "CharmmGuiFiles":
        """Construct from a CHARMM-GUI directory."""
        return cls(inputs_dir=Path(inputs_dir))

    @field_validator("inputs_dir", mode="after")
    @classmethod
    def _validate_inputs_root(cls, v: Path) -> Path:
        root = v.expanduser().resolve()

        if not root.is_dir():
            raise ValueError(f"inputs_dir is not a directory: {root}")

        missing_dirs = [
            root / name
            for name in cls.REQUIRED_DIRECTORIES
            if not (root / name).is_dir()
        ]
        missing_files = [
            root / name for name in cls.REQUIRED_FILES if not (root / name).is_file()
        ]

        if missing_dirs or missing_files:
            missing = [*missing_dirs, *missing_files]
            joined = "\n".join(f"  - {p}" for p in missing)
            raise ValueError(f"Missing required CHARMM-GUI files:\n{joined}")

        missing_parameter_files = [
            path for path in cls._parameter_file_paths(root) if not path.is_file()
        ]
        if missing_parameter_files:
            joined = "\n".join(f"  - {p}" for p in missing_parameter_files)
            raise ValueError(
                f"Missing topology/parameter files referenced by toppar.str:\n{joined}"
            )

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
            line = raw_line.split("!", maxsplit=1)[0].strip()
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
        return CharmmPsfFile(str(self.inputs_dir / "step5_assembly.psf"))

    # TODO: make this dynamic
    @property
    def pdb_file(self) -> PDBFile:
        return PDBFile(str(self.inputs_dir / "step5_assembly.pdb"))

    @property
    def params_file(self) -> CharmmParameterSet:
        # TODO: see what paths am I passing here
        print(*(str(path) for path in self._parameter_file_paths(self.inputs_dir)))
        return CharmmParameterSet(
            *(str(path) for path in self._parameter_file_paths(self.inputs_dir))
        )

    # TODO: there is an error here! I deleted this openmm inputs dir because I dont need it and is causing troubles now
    @property
    def openmm_inputs_dir(self) -> Path:
        openmm_dir = self.inputs_dir.parent / "openmm"
        if not openmm_dir.is_dir():
            raise FileNotFoundError(
                f"OpenMM protocol directory not found: {openmm_dir}"
            )
        return openmm_dir

    @property
    def box_lengths_angstrom(self) -> tuple[float, float, float]:
        values = self._assembly_box_values(self.inputs_dir / "step5_assembly.str")
        return values["A"], values["B"], values["C"]

    @staticmethod
    def _assembly_box_values(step5_assembly_file: Path) -> dict[str, float]:
        values: dict[str, float] = {}

        for raw_line in step5_assembly_file.read_text().splitlines():
            parts = raw_line.split()
            if len(parts) >= 4 and parts[0].upper() == "SET" and parts[2] == "=":
                key = parts[1].upper()
                if key in {"A", "B", "C"}:
                    values[key] = float(parts[3])

        missing = [key for key in ("A", "B", "C") if key not in values]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"Missing box dimensions ({joined}) in {step5_assembly_file}"
            )

        return values
