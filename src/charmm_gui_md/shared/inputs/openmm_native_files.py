import json
from pathlib import Path
from typing import ClassVar

from openmm.app import CharmmCrdFile, CharmmParameterSet, CharmmPsfFile, PDBFile
from pydantic import BaseModel, field_validator, model_validator

from charmm_gui_md.shared.profile import SystemProfile
from charmm_gui_md.shared.protocols import OpenMMStageProtocol


# TODO: This class should not be shared because the input files depend on the charmmgui system itself and
# will probably be different for membranes and solutions and others
class OpenMMNativeFiles(BaseModel):
    """Validated CHARMM-GUI OpenMM-native files needed for an OpenMM run."""

    model_config = {"arbitrary_types_allowed": True, "frozen": True}

    OPENMM_SUBDIRECTORY: ClassVar[str] = "openmm"

    inputs_dir: Path
    profile: SystemProfile

    @classmethod
    def from_root(
        cls,
        inputs_dir: Path,
        *,
        profile: SystemProfile,
    ) -> "OpenMMNativeFiles":
        return cls(
            inputs_dir=inputs_dir / cls.OPENMM_SUBDIRECTORY,
            profile=profile,
        )

    @field_validator("inputs_dir", mode="after")
    @classmethod
    def _resolve_inputs_dir(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @model_validator(mode="after")
    def _validate_inputs_root(self) -> "OpenMMNativeFiles":
        root = self.inputs_dir
        if not root.is_dir():
            raise ValueError(f"inputs_dir is not a directory: {root}")

        missing_files = [
            root / name
            for name in self._required_openmm_filenames()
            if not (root / name).is_file()
        ]
        if missing_files:
            joined = "\n".join(f"  - {path}" for path in missing_files)
            raise ValueError(f"Missing required OpenMM-native files:\n{joined}")

        missing_parameter_files = [
            path
            for path in self._parameter_paths_from_toppar_stream(root)
            if not path.is_file()
        ]
        if missing_parameter_files:
            joined = "\n".join(f"  - {path}" for path in missing_parameter_files)
            raise ValueError(
                "Missing topology/parameter files referenced by toppar.str:\n"
                f"{joined}"
            )

        missing_restraint_files = [
            path for path in self._required_restraint_paths() if not path.is_file()
        ]
        if missing_restraint_files:
            joined = "\n".join(f"  - {path}" for path in missing_restraint_files)
            raise ValueError(
                "Missing restraint files required by OpenMM-native protocols:\n"
                f"{joined}"
            )

        self._sysinfo_box_lengths(root / "sysinfo.dat")
        return self

    def _required_openmm_filenames(self) -> tuple[str, ...]:
        initial_files = tuple(
            f"{self.profile.initial_input_prefix}.{suffix}"
            for suffix in ("psf", "pdb", "crd")
        )
        stage_files = tuple(
            f"{stage_name}.inp"
            for stage_name in self.profile.protocol_schedule.stage_names
        )
        return (*initial_files, "sysinfo.dat", "toppar.str", *stage_files)

    def _required_restraint_paths(self) -> tuple[Path, ...]:
        restraints_dir = self.inputs_dir / "restraints"
        paths: list[Path] = []
        seen: set[Path] = set()

        for stage_name in self.profile.protocol_schedule.stage_names:
            protocol = OpenMMStageProtocol.from_file(
                step_name=stage_name,
                protocol_path=self.inputs_dir / f"{stage_name}.inp",
            )
            if not protocol.restraints_enabled:
                continue

            required_names: list[str] = []
            if (
                protocol.protein_backbone_restraint_kj_mol_nm2 > 0
                or protocol.protein_side_chain_restraint_kj_mol_nm2 > 0
            ):
                required_names.append("prot_pos.txt")
            if protocol.lipid_position_restraint_kj_mol_nm2 > 0:
                required_names.append("lipid_pos.txt")
            if protocol.lipid_dihedral_restraint_kj_mol_rad2 > 0:
                required_names.append("dihe.txt")

            for name in required_names:
                path = restraints_dir / name
                if path not in seen:
                    paths.append(path)
                    seen.add(path)

        return tuple(paths)

    @classmethod
    def _parameter_paths_from_toppar_stream(cls, root: Path) -> tuple[Path, ...]:
        paths: list[Path] = []
        seen: set[Path] = set()

        for raw_line in (root / "toppar.str").read_text().splitlines():
            line = raw_line.split("!", maxsplit=1)[0].split("#", maxsplit=1)[0].strip()
            if not line:
                continue

            reference = cls._parameter_reference(line)
            path = cls._resolve_parameter_reference(root, reference)
            if path not in seen:
                paths.append(path)
                seen.add(path)

        return tuple(paths)

    @staticmethod
    def _parameter_reference(line: str) -> str:
        words = line.split()
        if len(words) != 1:
            raise ValueError(
                f"Unsupported OpenMM-native toppar.str entry: {line!r}. "
                "Expected one parameter file path per line."
            )
        return words[0]

    @staticmethod
    def _resolve_parameter_reference(root: Path, reference: str) -> Path:
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
        return CharmmPsfFile(
            str(self.inputs_dir / f"{self.profile.initial_input_prefix}.psf")
        )

    @property
    def pdb_file(self) -> PDBFile:
        return PDBFile(str(self.initial_coordinates_path))

    @property
    def crd_file(self) -> CharmmCrdFile:
        return CharmmCrdFile(
            str(self.inputs_dir / f"{self.profile.initial_input_prefix}.crd")
        )

    @property
    def restraint_reference_positions(self):
        return self.crd_file.positions

    @property
    def initial_coordinates_path(self) -> Path:
        return self.inputs_dir / f"{self.profile.initial_input_prefix}.pdb"

    @property
    def params_file(self) -> CharmmParameterSet:
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
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {sysinfo_file}") from exc

        dimensions = data.get("dimensions")
        if not isinstance(dimensions, (list, tuple)) or len(dimensions) < 3:
            raise ValueError(f"Missing dimensions[0:3] in {sysinfo_file}")

        try:
            return float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid box dimensions in {sysinfo_file}") from exc
