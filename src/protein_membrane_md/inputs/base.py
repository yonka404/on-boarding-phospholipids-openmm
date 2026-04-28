from pathlib import Path
from typing import Protocol

from openmm.app import CharmmParameterSet, CharmmPsfFile, PDBFile


class SimulationInputFiles(Protocol):
    inputs_dir: Path

    @property
    def psf_file(self) -> CharmmPsfFile: ...

    @property
    def pdb_file(self) -> PDBFile: ...

    @property
    def params_file(self) -> CharmmParameterSet: ...

    @property
    def box_lengths_angstrom(self) -> tuple[float, float, float]: ...
