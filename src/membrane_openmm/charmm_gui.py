import re
from pathlib import Path
from typing import ClassVar

from openmm.app import CharmmParameterSet, CharmmPsfFile, PDBFile
from openmm.unit import angstrom, degree
from pydantic import BaseModel, field_validator


class CharmmGuiFiles(BaseModel):
    """Validated existence of CHARMM-GUI input directory ."""

    model_config = {"frozen": True}

    REQUIRED_FILES: ClassVar[tuple[str, ...]] = (
        "membrane_restraint.charmm_openmm.str",
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

    # TODO: think if this is necessary
    # @property
    # def psf_path(self) -> Path:
    #     return self.inputs_dir / "step5_assembly.psf"
    #
    # @property
    # def pdb_path(self) -> Path:
    #     return self.inputs_dir / "step5_assembly.pdb"
    #
    # @property
    # def box_path(self) -> Path:
    #     return self.inputs_dir / "step5_assembly.str"
    #
    # @property
    # def toppar_str_path(self) -> Path:
    #     return self.inputs_dir / "toppar.str"


class CharmmGuiSystem(BaseModel):
    """Validated CHARMM-GUI system with metadata parsed from input files (the CharmmGuiFiles)."""

    model_config = {"frozen": True}

    files: CharmmGuiFiles

    boxtype: str = "RECT"
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    zcen: float = 0.0
    nliptop: int
    nlipbot: int
    nwater: int
    niontot: int

    @classmethod
    def from_files(cls, files: CharmmGuiFiles) -> "CharmmGuiSystem":
        values = cls._parse_step_assembly_file(files)
        return cls(files=files, **values)

    # @field_validator("boxtype")
    # @classmethod
    # def _normalize_boxtype(cls, v: str) -> str:
    #     value = v.strip().upper()
    #     if not value:
    #         raise ValueError("boxtype must not be empty")
    #     return value
    #
    # @field_validator("a", "b", "c")
    # @classmethod
    # def _box_dims_positive(cls, v: float, info) -> float:
    #     if v <= 0:
    #         raise ValueError(
    #             f"Box dimension '{info.field_name}' must be positive, got {v}"
    #         )
    #     return v
    #
    # @field_validator("alpha", "beta", "gamma")
    # @classmethod
    # def _angles_in_range(cls, v: float, info) -> float:
    #     if not (0.0 < v < 180.0):
    #         raise ValueError(f"Angle '{info.field_name}' must be in (0, 180), got {v}")
    #     return v
    #
    # @field_validator("nliptop", "nlipbot", "nwater", "niontot")
    # @classmethod
    # def _counts_non_negative(cls, v: int, info) -> int:
    #     if v < 0:
    #         raise ValueError(f"'{info.field_name}' must be >= 0, got {v}")
    #     return v
    #
    # @property
    # def total_lipids(self) -> int:
    #     return self.nliptop + self.nlipbot
