import logging

from openmm.app import Simulation

from charmm_gui_md.shared.artifacts import StageArtifacts
from charmm_gui_md.shared.protocols import OpenMMStageProtocol

logger = logging.getLogger(__name__)


class StageCheckpointManager:
    def restore(
        self,
        simulation: Simulation,
        artifacts: StageArtifacts,
        protocol: OpenMMStageProtocol,
    ) -> int | None:
        if not artifacts.checkpoint_path.is_file():
            return None

        try:
            simulation.loadCheckpoint(str(artifacts.checkpoint_path))
        except Exception as exc:
            raise RuntimeError(
                f"[{protocol.step_name}] Could not load checkpoint "
                f"{artifacts.checkpoint_path}. Existing outputs were left unchanged."
            ) from exc

        completed_steps = simulation.currentStep
        if completed_steps < 0:
            raise RuntimeError(
                f"[{protocol.step_name}] Cannot resume: checkpoint step "
                f"{completed_steps} is invalid. Existing outputs were left unchanged."
            )
        if completed_steps > protocol.dynamics_steps:
            raise RuntimeError(
                f"[{protocol.step_name}] Cannot resume: checkpoint step "
                f"{completed_steps} exceeds configured stage target "
                f"{protocol.dynamics_steps}. Existing outputs were left unchanged."
            )

        remaining_steps = protocol.dynamics_steps - completed_steps
        logger.info(
            "[%s] Loaded checkpoint %s at step %d; %d steps remain",
            protocol.step_name,
            artifacts.checkpoint_path,
            completed_steps,
            remaining_steps,
        )
        return remaining_steps

    def clear(
        self,
        artifacts: StageArtifacts,
        protocol: OpenMMStageProtocol,
    ) -> None:
        artifacts.checkpoint_path.unlink(missing_ok=True)
        logger.info(
            "[%s] Removed completed recovery checkpoint %s",
            protocol.step_name,
            artifacts.checkpoint_path,
        )
