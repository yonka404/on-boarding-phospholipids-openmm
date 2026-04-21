import pdb
from pathlib import Path

from openmm import LangevinMiddleIntegrator
from openmm.app import PME, HBonds, Simulation
from openmm.unit import kelvin, kilojoule_per_mole, nanometer, picosecond

from membrane_openmm.charmm_gui import CharmmGuiFiles


def run_single_step(
    inputs_dir: Path,
    outputs_dir: Path,
    step_name: str,
) -> None:

    files = CharmmGuiFiles.from_root(inputs_dir=inputs_dir)

    psf = files.psf_file
    pdb = files.pdb_file
    params = files.params_file

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
    # TODO: apply restraints
    simulation.minimizeEnergy(
        tolerance=100.0 * kilojoule_per_mole / nanometer,
        maxIterations=5000,
    )
    print("Done.")
