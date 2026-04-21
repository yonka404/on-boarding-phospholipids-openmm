import re
from pathlib import Path
from typing import ClassVar

from openmm.app import CharmmParameterSet, CharmmPsfFile, PDBFile
from openmm.unit import angstrom, degree
from pydantic import BaseModel, field_validator


class CharmmGuiFiles(BaseModel):
    """Validated CHARMM-GUI input directory with canonical file accessors."""

    model_config = {"frozen": True}

    REQUIRED_FILES: ClassVar[tuple[str, ...]] = (
        "membrane_restraint.charmm_openmm.str",
        "step6.1_equilibration.inp",
        "step6.2_equlibration.inp",
        "step6.3_equlibration.inp",
        "step6.4_equlibration.inp",
        "step6.5_equlibration.inp",
        "step6.6_equlibration.inp",
        "step6.7_equlibration.inp",
        "step7_production.inp",
        "toppar.str",
    )

    inputs_root: Path

    @classmethod
    def from_root(cls, inputs_root: Path) -> "CharmmGuiFiles":
        """Construct from a CHARMM-GUI directory."""
        return cls(inputs_root=Path(inputs_root))

    @field_validator("inputs_root", mode="after")
    @classmethod
    def _validate_inputs_root(cls, v: Path) -> Path:
        root = v.expanduser().resolve()

        if not root.is_dir():
            raise ValueError(f"inputs_root is not a directory: {root}")

        missing = [
            root / name for name in cls.REQUIRED_FILES if not (root / name).is_file()
        ]
        if missing:
            joined = "\n".join(f"  - {p}" for p in missing)
            raise ValueError(f"Missing required CHARMM-GUI files:\n{joined}")

        return root

    @property
    def psf_path(self) -> Path:
        return self.inputs_root / "step5_assembly.psf"

    @property
    def pdb_path(self) -> Path:
        return self.inputs_root / "step5_assembly.pdb"

    @property
    def box_path(self) -> Path:
        return self.inputs_root / "step5_assembly.str"

    @property
    def toppar_str_path(self) -> Path:
        return self.inputs_root / "toppar.str"


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
        values = cls._parse_step_assembly_file(files.box_path)
        return cls(files=files, **values)

    @field_validator("boxtype")
    @classmethod
    def _normalize_boxtype(cls, v: str) -> str:
        value = v.strip().upper()
        if not value:
            raise ValueError("boxtype must not be empty")
        return value

    @field_validator("a", "b", "c")
    @classmethod
    def _box_dims_positive(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(
                f"Box dimension '{info.field_name}' must be positive, got {v}"
            )
        return v

    @field_validator("alpha", "beta", "gamma")
    @classmethod
    def _angles_in_range(cls, v: float, info) -> float:
        if not (0.0 < v < 180.0):
            raise ValueError(f"Angle '{info.field_name}' must be in (0, 180), got {v}")
        return v

    @field_validator("nliptop", "nlipbot", "nwater", "niontot")
    @classmethod
    def _counts_non_negative(cls, v: int, info) -> int:
        if v < 0:
            raise ValueError(f"'{info.field_name}' must be >= 0, got {v}")
        return v

    @property
    def total_lipids(self) -> int:
        return self.nliptop + self.nlipbot

    def load_params(self) -> CharmmParameterSet:
        param_paths = [
            str((self.files.inputs_root / rel_path).resolve())
            for rel_path in _parse_toppar_stream(self.files.toppar_str_path)
        ]
        return CharmmParameterSet(*param_paths)

    def load_psf(self) -> CharmmPsfFile:
        psf = CharmmPsfFile(str(self.files.psf_path))
        psf.setBox(
            self.a * angstrom,
            self.b * angstrom,
            self.c * angstrom,
            self.alpha * degree,
            self.beta * degree,
            self.gamma * degree,
        )
        return psf

    def load_pdb(self) -> PDBFile:
        return PDBFile(str(self.files.pdb_path))

    def load(self) -> tuple[CharmmPsfFile, PDBFile, CharmmParameterSet]:
        """
        Convenience loader for downstream simulation setup.
        Returns (psf, pdb, params).
        """
        psf = self.load_psf()
        pdb = self.load_pdb()
        params = self.load_params()
        return psf, pdb, params

    @staticmethod
    def _parse_step_assembly_file(path: Path) -> dict[str, str | float | int]:
        text = _read_text(path)
        values: dict[str, str] = {}

        for line in text.splitlines():
            line = line.strip()
            if not line or not line.upper().startswith("SET "):
                continue
            match = re.match(r"SET\s+(\w+)\s*=\s*(.+)", line, flags=re.IGNORECASE)
            if match:
                key, value = match.groups()
                values[key.upper()] = value.strip()

        def f(key: str, default: float = 0.0) -> float:
            return float(values.get(key, default))

        def i(key: str, default: int = 0) -> int:
            return int(float(values.get(key, default)))

        return {
            "boxtype": values.get("BOXTYPE", "RECT"),
            "a": f("A"),
            "b": f("B"),
            "c": f("C"),
            "alpha": f("ALPHA", 90.0),
            "beta": f("BETA", 90.0),
            "gamma": f("GAMMA", 90.0),
            "zcen": f("ZCEN", 0.0),
            "nliptop": i("NLIPTOP"),
            "nlipbot": i("NLIPBOT"),
            "nwater": i("NWATER"),
            "niontot": i("NIONTOT"),
        }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_toppar_stream(toppar_str: Path) -> list[str]:
    """Parse CHARMM-GUI toppar.str into an ordered file list for CharmmParameterSet."""
    files: list[str] = []
    pattern_open = re.compile(r"name\s+(toppar/\S+)", flags=re.IGNORECASE)
    pattern_stream = re.compile(r"stream\s+(toppar/\S+)", flags=re.IGNORECASE)

    for raw_line in _read_text(toppar_str).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!") or line.startswith("*"):
            continue

        m_open = pattern_open.search(line)
        if m_open:
            files.append(m_open.group(1))
            continue

        m_stream = pattern_stream.search(line)
        if m_stream:
            files.append(m_stream.group(1))

    if not files:
        raise ValueError(
            f"Could not parse any topology/parameter files from {toppar_str}"
        )

    return files
