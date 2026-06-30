import logging

from openmm.app import CheckpointReporter, DCDReporter, Simulation, StateDataReporter

from charmm_gui_md.shared.artifacts import StageArtifacts
from charmm_gui_md.shared.protocols import OpenMMStageProtocol

logger = logging.getLogger(__name__)


class StageReporterInstaller:
    def install(
        self,
        simulation: Simulation,
        artifacts: StageArtifacts,
        protocol: OpenMMStageProtocol,
        *,
        append: bool = False,
    ) -> None:
        if append:
            required_outputs = (artifacts.state_data_path, artifacts.trajectory_path)
            missing_outputs = [path for path in required_outputs if not path.is_file()]
            if missing_outputs:
                missing_list = ", ".join(str(path) for path in missing_outputs)
                raise FileNotFoundError(
                    f"[{protocol.step_name}] Cannot resume without existing reporter "
                    f"outputs: {missing_list}"
                )

        simulation.reporters.append(
            StateDataReporter(
                str(artifacts.state_data_path),
                protocol.state_report_interval_steps,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
                separator=",",
                append=append,
            )
        )

        simulation.reporters.append(
            DCDReporter(
                str(artifacts.trajectory_path),
                protocol.trajectory_report_interval_steps,
                append=append,
                enforcePeriodicBox=True,
            )
        )

        simulation.reporters.append(
            CheckpointReporter(
                str(artifacts.checkpoint_path),
                protocol.trajectory_report_interval_steps,
            )
        )

        logger.info(
            "[%s] Reporting state data to %s, trajectory to %s, and checkpoints to %s",
            protocol.step_name,
            artifacts.state_data_path,
            artifacts.trajectory_path,
            artifacts.checkpoint_path,
        )
