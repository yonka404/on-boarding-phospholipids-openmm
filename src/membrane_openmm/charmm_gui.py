from pathlib import Path
from typing import ClassVar

from openmm.app import CharmmParameterSet, CharmmPsfFile, PDBFile
from pydantic import BaseModel, field_validator


class CharmmGuiFiles(BaseModel):
    """Validated existence of CHARMM-GUI input directory ."""

    model_config = {"frozen": True}

    REQUIRED_FILES: ClassVar[tuple[str, ...]] = (
        "membrane_restraint.charmm_openmm.str",
        "step5_assembly.psf",
        "step5_assembly.pdb",
        "step5_assembly.str",
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

        missing = [
            root / name for name in cls.REQUIRED_FILES if not (root / name).is_file()
        ]
        if missing:
            joined = "\n".join(f"  - {p}" for p in missing)
            raise ValueError(f"Missing required CHARMM-GUI files:\n{joined}")

        return root

    @property
    def psf_file(self) -> CharmmPsfFile:
        return CharmmPsfFile(str(self.inputs_dir / "step5_assembly.psf"))

    @property
    def pdb_file(self) -> PDBFile:
        return PDBFile(str(self.inputs_dir / "step5_assembly.pdb"))

    @property
    def params_file(self) -> CharmmParameterSet:
        return CharmmParameterSet(str(self.inputs_dir / "step5_assembly.str"))
