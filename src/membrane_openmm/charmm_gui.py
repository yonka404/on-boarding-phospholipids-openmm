from dataclasses import dataclass
from pathlib import Path
import re

from openmm.app import CharmmParameterSet, CharmmPsfFile, PDBFile
from openmm.unit import angstrom, degree


@dataclass(frozen=True)
class SystemMetadata:
    boxtype: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    zcen: float
    nliptop: int
    nlipbot: int
    nwater: int
    niontot: int

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


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_step5_assembly_str(path: Path) -> SystemMetadata:
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


def parse_toppar_stream(toppar_str: Path) -> list[str]:
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
        raise ValueError(f"Could not parse any topology/parameter files from {toppar_str}")
    return files


def load_charmm_gui_system(system_root: str | Path) -> LoadedCharmmGuiSystem:
    root = Path(system_root).expanduser().resolve()

    # TODO: Would be nice that Pydantic validates these files before anything else
    psf_path = root / "step5_assembly.psf"
    pdb_path = root / "step5_assembly.pdb"
    box_path = root / "step5_assembly.str"
    toppar_str = root / "toppar.str"

    # TODO: Instead of having this here pointed as missing files
    missing = [p for p in [psf_path, pdb_path, box_path, toppar_str] if not p.exists()]
    if missing:
        joined = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(f"Missing required CHARMM-GUI files:\n{joined}")

    metadata = parse_step5_assembly_str(box_path)

    param_paths = [str((root / rel_path).resolve()) for rel_path in parse_toppar_stream(toppar_str)]
    params = CharmmParameterSet(*param_paths)
    psf = CharmmPsfFile(str(psf_path))
    psf.setBox(
        metadata.a * angstrom,
        metadata.b * angstrom,
        metadata.c * angstrom,
        metadata.alpha * degree,
        metadata.beta * degree,
        metadata.gamma * degree,
    )
    pdb = PDBFile(str(pdb_path))

    return LoadedCharmmGuiSystem(
        system_root=root,
        psf=psf,
        pdb=pdb,
        params=params,
        metadata=metadata,
    )
