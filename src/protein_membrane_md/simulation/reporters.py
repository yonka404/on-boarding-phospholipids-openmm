import logging

from openmm.app import DCDReporter, Simulation, StateDataReporter

from protein_membrane_md.artifacts import StageArtifacts
from protein_membrane_md.protocols import OpenMMStageProtocol

logger = logging.getLogger(__name__)


class StageReporterInstaller:
    def install(
        self,
        simulation: Simulation,
        artifacts: StageArtifacts,
        protocol: OpenMMStageProtocol,
    ) -> None:
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
            )
        )

        simulation.reporters.append(
            DCDReporter(
                str(artifacts.trajectory_path),
                protocol.trajectory_report_interval_steps,
            )
        )

        logger.info(
            "[%s] Reporting state data to %s and trajectory to %s",
            protocol.step_name,
            artifacts.state_data_path,
            artifacts.trajectory_path,
        )
