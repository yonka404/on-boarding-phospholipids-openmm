from pathlib import Path

from openmm import LangevinMiddleIntegrator
from openmm.app import Simulation
from openmm.unit import kelvin, kilojoule_per_mole, nanometer, picosecond

from membrane_openmm.charmm_gui import CharmmGuiFiles


def run_single_step(
    inputs_dir: Path,
    outputs_dir: Path,
    step_name: str,
    starting_pdb: Path | None = None,
) -> None:
    files = CharmmGuiFiles.from_root(inputs_dir=inputs_dir)

    step_output_dir = outputs_dir / step_name
    step_output_dir.mkdir(parents=True, exist_ok=True)

    # Load the common system definition
    psf = files.psf_file
    params = files.params_file

    # Choose coordinates: original structure or previous stage output
    if starting_pdb is None:
        coordinates_pdb = files.pdb_file
        print(
            f"[{step_name}] Using initial CHARMM-GUI coordinates: {inputs_dir / 'step5_assembly.pdb'}"
        )
    else:
        if not starting_pdb.is_file():
            raise FileNotFoundError(f"Starting PDB not found: {starting_pdb}")
        coordinates_pdb = PDBFile(str(starting_pdb))
        print(f"[{step_name}] Using chained coordinates: {starting_pdb}")

    # Create the OpenMM system
    system = psf.createSystem(params)

    # Define the integrator
    integrator = LangevinMiddleIntegrator(
        303.15 * kelvin,
        1.0 / picosecond,
        0.001 * picosecond,
    )

    # Build the simulation
    simulation = Simulation(psf.topology, system, integrator)
    simulation.context.setPositions(coordinates_pdb.positions)

    print(f"[{step_name}] Minimizing...")
    simulation.minimizeEnergy(
        tolerance=100.0 * kilojoule_per_mole / nanometer,
        maxIterations=5000,
    )
    print(f"[{step_name}] Done.")

    output_pdb = step_output_dir / "minimized.pdb"
    state = simulation.context.getState(getPositions=True)

    with open(output_pdb, "w") as handle:
        PDBFile.writeFile(simulation.topology, state.getPositions(), handle)

    print(f"[{step_name}] Wrote minimized structure to: {output_pdb}")

    return output_pdb
