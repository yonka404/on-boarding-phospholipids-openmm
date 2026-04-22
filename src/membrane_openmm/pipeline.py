from dataclasses import dataclass
import logging
from pathlib import Path

from openmm import LangevinMiddleIntegrator
from openmm.app import PDBFile, Simulation
from openmm.unit import kelvin, kilojoule_per_mole, nanometer, picosecond

from membrane_openmm.charmm_gui import CharmmGuiFiles

logger = logging.getLogger(__name__)

PROTOCOL_STAGES = (
    "step6.1_equilibration",
    "step6.2_equilibration",
    "step6.3_equilibration",
    "step6.4_equilibration",
    "step6.5_equilibration",
    "step6.6_equilibration",
    "step7_production",
)

FINAL_COORDINATES_FILENAME = "final_coordinates.pdb"


@dataclass(frozen=True)
class CoordinateSource:
    path: Path
    description: str


def _resolve_coordinate_source(
    inputs_dir: Path,
    outputs_dir: Path,
    step_name: str,
) -> CoordinateSource:
    if step_name not in PROTOCOL_STAGES:
        supported_steps = ", ".join(PROTOCOL_STAGES)
        raise ValueError(
            f"Unsupported step_name {step_name!r}. Expected one of: {supported_steps}"
        )

    step_index = PROTOCOL_STAGES.index(step_name)
    if step_index == 0:
        return CoordinateSource(
            path=inputs_dir / "step5_assembly.pdb",
            description="initial CHARMM-GUI coordinates",
        )

    previous_step = PROTOCOL_STAGES[step_index - 1]
    return CoordinateSource(
        path=outputs_dir / previous_step / FINAL_COORDINATES_FILENAME,
        description=f"coordinates from previous protocol stage '{previous_step}'",
    )


def _load_coordinates(source: CoordinateSource) -> PDBFile:
    if not source.path.is_file():
        raise FileNotFoundError(
            f"Coordinate file not found for {source.description}: {source.path}"
        )

    return PDBFile(str(source.path))


def run_single_step(
    inputs_dir: Path,
    outputs_dir: Path,
    step_name: str,
) -> Path:
    files = CharmmGuiFiles.from_root(inputs_dir=inputs_dir)
    coordinate_source = _resolve_coordinate_source(
        inputs_dir=files.inputs_dir,
        outputs_dir=outputs_dir,
        step_name=step_name,
    )
    coordinates_pdb = _load_coordinates(coordinate_source)

    step_output_dir = outputs_dir / step_name
    step_output_dir.mkdir(parents=True, exist_ok=True)

    psf = files.psf_file
    params = files.params_file

    logger.info(
        "[%s] Using %s: %s",
        step_name,
        coordinate_source.description,
        coordinate_source.path,
    )

    system = psf.createSystem(params)

    integrator = LangevinMiddleIntegrator(
        303.15 * kelvin,
        1.0 / picosecond,
        0.001 * picosecond,
    )

    simulation = Simulation(psf.topology, system, integrator)
    simulation.context.setPositions(coordinates_pdb.positions)

    logger.info("[%s] Minimizing", step_name)
    simulation.minimizeEnergy(
        tolerance=100.0 * kilojoule_per_mole / nanometer,
        maxIterations=5000,
    )
    logger.info("[%s] Minimization complete", step_name)

    output_pdb = step_output_dir / FINAL_COORDINATES_FILENAME
    state = simulation.context.getState(getPositions=True)

    with open(output_pdb, "w") as handle:
        PDBFile.writeFile(simulation.topology, state.getPositions(), handle)

    logger.info("[%s] Wrote output coordinates to %s", step_name, output_pdb)

    return output_pdb
