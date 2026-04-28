import logging

from openmm.app import PDBFile, Simulation

from protein_membrane_md.artifacts import StageArtifacts
from protein_membrane_md.protocols import OpenMMStageProtocol

logger = logging.getLogger(__name__)


class StageOutputWriter:
    def write(
        self,
        simulation: Simulation,
        artifacts: StageArtifacts,
        protocol: OpenMMStageProtocol,
    ):
        simulation.saveState(str(artifacts.final_state_path))

        state = simulation.context.getState(getPositions=True)
        simulation.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())

        with open(artifacts.final_coordinates_path, "w") as handle:
            PDBFile.writeFile(
                simulation.topology,
                state.getPositions(),
                handle,
            )

        logger.info(
            "[%s] Wrote restart state to %s",
            protocol.step_name,
            artifacts.final_state_path,
        )
        logger.info(
            "[%s] Wrote final coordinates to %s",
            protocol.step_name,
            artifacts.final_coordinates_path,
        )

        return artifacts.final_coordinates_path
