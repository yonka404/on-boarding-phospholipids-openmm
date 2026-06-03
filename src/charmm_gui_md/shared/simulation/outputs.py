import logging

from openmm.app import PDBFile, Simulation

from charmm_gui_md.shared.artifacts import StageArtifacts
from charmm_gui_md.shared.protocols import OpenMMStageProtocol

logger = logging.getLogger(__name__)


class StageOutputWriter:
    def write(
        self,
        simulation: Simulation,
        artifacts: StageArtifacts,
        protocol: OpenMMStageProtocol,
    ):
        simulation.saveState(str(artifacts.final_state_path))

        state = simulation.context.getState(
            getPositions=True,
            enforcePeriodicBox=True,
        )
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
