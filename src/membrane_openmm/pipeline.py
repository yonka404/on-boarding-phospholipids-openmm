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

    # TODO: apply restraints
    psf, pdb, params = system_info.load()

    system = psf.createSystem(
        params,
        nonbondedMethod=PME,
        nonbondedCutoff=1.2 * nanometer,
        constraints=HBonds,
    )

    integrator = LangevinMiddleIntegrator(
        303.15 * kelvin,
        1.0 / picosecond,
        0.001 * picosecond,
    )

    simulation = Simulation(psf.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    print("Minimizing...")
    simulation.minimizeEnergy(
        tolerance=100.0 * kilojoule_per_mole / nanometer,
        maxIterations=5000,
    )
    print("Done.")
