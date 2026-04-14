import re
from dataclasses import dataclass
from pathlib import Path

from openmm.app import CharmmParameterSet, CharmmPsfFile, PDBFile
from openmm.unit import angstrom, degree
from pydantic import BaseModel, field_validator, model_validator


class CharmmGuiFiles(BaseModel):
    """Validated set of CHARMM-GUI input file paths. All must exist."""

    model_config = {"frozen": True}

    system_root: Path
    psf_path: Path
    pdb_path: Path
    box_path: Path
    toppar_str_path: Path

    @classmethod
    def from_root(cls, system_root: str | Path) -> "CharmmGuiFiles":
        """Construct from a CHARMM-GUI directory, resolving canonical file names."""
        root = Path(system_root).expanduser().resolve()
        return cls(
            system_root=root,
            psf_path=root / "step5_assembly.psf",
            pdb_path=root / "step5_assembly.pdb",
            box_path=root / "step5_assembly.str",
            toppar_str_path=root / "toppar.str",
        )

    @field_validator("system_root")
    @classmethod
    def _root_must_be_dir(cls, v: Path) -> Path:
        v = v.expanduser().resolve()
        if not v.is_dir():
            raise ValueError(f"system_root is not a directory: {v}")
        return v

    @model_validator(mode="after")
    def _all_files_must_exist(self) -> "CharmmGuiFiles":
        missing = []

        for field_name in ("psf_path", "pdb_path", "box_path", "toppar_str_path"):
            p = getattr(self, field_name)
            if not p.exists():
                missing.append(p)
        if missing:
            joined = "\n".join(f"  - {p}" for p in missing)
            raise ValueError(f"Missing required CHARMM-GUI files:\n{joined}")
        return self


class SystemMetadata(BaseModel):
    """Box and composition metadata parsed from step5_assembly.str."""

    model_config = {"frozen": True}

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

    @field_validator("nliptop", "nlipbot")
    @classmethod
    def _lipid_counts_non_negative(cls, v: int, info) -> int:
        if v < 0:
            raise ValueError(f"'{info.field_name}' must be >= 0, got {v}")
        return v

    @property
    def total_lipids(self) -> int:
        return self.nliptop + self.nlipbot


@dataclass(frozen=True)
class LoadedCharmmGuiSystem:
    system_root: Path
    psf: CharmmPsfFile
    pdb: PDBFile
    params: CharmmParameterSet
    metadata: SystemMetadata


# --------------------------------------------------------------
# --------------------------------------------------------------


class CharmGuiSystem(BaseModel):
    """CHARMM-GUI system representation, with validation and loading logic."""

    model_config = {"frozen": True}

    files: CharmmGuiFiles

    def load(self) -> LoadedCharmmGuiSystem:
        """Load a CHARMM-GUI system. Expects a pre-validated CharmmGuiFiles object."""

        metadata = self._parse_step_assembly_file()

        param_paths = [
            str((self.files.system_root / rel_path).resolve())
            for rel_path in _parse_toppar_stream(self.files.toppar_str_path)
        ]

        params = CharmmParameterSet(*param_paths)
        psf = CharmmPsfFile(str(files.psf_path))
        psf.setBox(
            metadata.a * angstrom,
            metadata.b * angstrom,
            metadata.c * angstrom,
            metadata.alpha * degree,
            metadata.beta * degree,
            metadata.gamma * degree,
        )
        pdb = PDBFile(str(files.pdb_path))

        return LoadedCharmmGuiSystem(
            system_root=files.system_root,
            psf=psf,
            pdb=pdb,
            params=params,
            metadata=metadata,
        )

    def _parse_step_assembly_file(self) -> SystemMetadata:
        text = _read_text(self.files.box_path)
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

        return SystemMetadata(
            boxtype=values.get("BOXTYPE", "RECT"),
            a=f("A"),
            b=f("B"),
            c=f("C"),
            alpha=f("ALPHA", 90.0),
            beta=f("BETA", 90.0),
            gamma=f("GAMMA", 90.0),
            zcen=f("ZCEN", 0.0),
            nliptop=i("NLIPTOP"),
            nlipbot=i("NLIPBOT"),
            nwater=i("NWATER"),
            niontot=i("NIONTOT"),
        )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_toppar_stream(toppar_str: Path) -> list[str]:
    """Parse CHARMM-GUI toppar.str into an ordered file list for CharmmParameterSet.

    We keep the exact stream/open order from CHARMM-GUI instead of glob-sorting,
    because parameter order can matter for CHARMM-style inputs.
    """
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
