import logging

from openmm.app import PDBFile, Simulation
from openmm.unit import kelvin

from protein_membrane_md.protocols import OpenMMStageProtocol

logger = logging.getLogger(__name__)


class SimulationInitializer:
    def initialize(
        self,
        simulation: Simulation,
        restart_source,
        protocol: OpenMMStageProtocol,
    ) -> None:
        if restart_source.state_path is not None:
            logger.info(
                "[%s] Loading restart state from %s",
                protocol.step_name,
                restart_source.state_path,
            )
            simulation.loadState(str(restart_source.state_path))
            return

        if not restart_source.coordinates_path.is_file():
            raise FileNotFoundError(
                f"[{protocol.step_name}] Coordinate file not found for "
                f"{restart_source.description}: {restart_source.coordinates_path}"
            )

        coordinates_pdb = PDBFile(str(restart_source.coordinates_path))
        simulation.context.setPositions(coordinates_pdb.positions)

        logger.info(
            "[%s] Loaded coordinates from %s: %s",
            protocol.step_name,
            restart_source.description,
            restart_source.coordinates_path,
        )

        self._initialize_velocities(simulation, protocol)

    def _initialize_velocities(
        self,
        simulation: Simulation,
        protocol: OpenMMStageProtocol,
    ) -> None:
        target_temperature = (
            protocol.velocity_temperature_kelvin
            if protocol.generate_velocities
            and protocol.velocity_temperature_kelvin is not None
            else protocol.temperature_kelvin
        )

        simulation.context.setVelocitiesToTemperature(target_temperature * kelvin)

        if protocol.generate_velocities:
            logger.info(
                "[%s] Generated initial velocities at %.2f K",
                protocol.step_name,
                target_temperature,
            )
        else:
            logger.warning(
                "[%s] No restart state was available, so velocities were regenerated at %.2f K",
                protocol.step_name,
                target_temperature,
            )
