from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from openmm import Platform
from openmm.app import DCDReporter
from openmm.unit import (
    MOLAR_GAS_CONSTANT_R,
    kelvin,
    kilojoule_per_mole,
    nanometer,
)

from membrane_openmm.charmm_gui import CharmmGuiFiles, CharmmGuiSystem


def run_case(
    inputs_root: Path,
    outdir: str | Path,
    temperature_k: float = 303.15,
) -> None:

    files = CharmmGuiFiles.from_root(inputs_root=inputs_root)
    system = CharmmGuiSystem.from_files(files=files)
    loaded = system.load()

    # TODO: I think we should use here the validated output dir now
    _write_metadata(outdir, loaded, temperature_k)
