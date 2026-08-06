import logging
from pathlib import Path

from openmm.unit import kilojoule_per_mole, nanometer

from charmm_gui_md.shared.artifacts import RestartResolver, StageArtifacts
from charmm_gui_md.shared.inputs.openmm_native_files import OpenMMNativeFiles
from charmm_gui_md.shared.profile import SystemProfile
from charmm_gui_md.shared.protocols import OpenMMStageProtocol
from charmm_gui_md.shared.simulation import (
    OpenMMSimulationFactory,
    SimulationInitializer,
    StageCheckpointManager,
    StageOutputWriter,
    StageReporterInstaller,
)

logger = logging.getLogger(__name__)


class StageRunner:
    def __init__(
        self,
        profile: SystemProfile,
        restart_resolver: RestartResolver | None = None,
        simulation_factory: OpenMMSimulationFactory | None = None,
        simulation_initializer: SimulationInitializer | None = None,
        reporter_installer: StageReporterInstaller | None = None,
        output_writer: StageOutputWriter | None = None,
        checkpoint_manager: StageCheckpointManager | None = None,
    ) -> None:
        self.profile = profile
        self.restart_resolver = restart_resolver or RestartResolver(profile=profile)
        self.simulation_factory = simulation_factory or OpenMMSimulationFactory()
        self.simulation_initializer = simulation_initializer or SimulationInitializer()
        self.reporter_installer = reporter_installer or StageReporterInstaller()
        self.output_writer = output_writer or StageOutputWriter()
        self.checkpoint_manager = checkpoint_manager or StageCheckpointManager()

    def run(
        self,
        inputs_dir: Path,
        outputs_dir: Path,
        step_name: str,
    ) -> Path:
        self.profile.protocol_schedule.require_stage(step_name)

        files = OpenMMNativeFiles.from_root(
            inputs_dir=inputs_dir,
            profile=self.profile,
        )

        protocol = OpenMMStageProtocol.from_file(
            step_name=step_name,
            protocol_path=files.inputs_dir / f"{step_name}.inp",
        )
        artifacts = StageArtifacts.for_stage(outputs_dir, step_name)
        artifacts.create_directory()

        simulation = self.simulation_factory.create(files, protocol)
        checkpoint_remaining_steps = self.checkpoint_manager.restore(
            simulation,
            artifacts,
            protocol,
        )

        if checkpoint_remaining_steps is None:
            resumed = False
            restart_source = self.restart_resolver.resolve(
                inputs_dir=files.inputs_dir,
                outputs_dir=outputs_dir,
                step_name=step_name,
                initial_coordinates_path=files.initial_coordinates_path,
            )
            needs_velocity_initialization = restart_source.state_path is None
            self.simulation_initializer.initialize(simulation, restart_source, protocol)
            remaining_steps = protocol.dynamics_steps
        else:
            resumed = True
            needs_velocity_initialization = False
            remaining_steps = checkpoint_remaining_steps

        if not resumed and protocol.has_minimization:
            logger.info(
                "[%s] Minimizing with tolerance %.2f kJ/mol/nm for up to %d iterations",
                protocol.step_name,
                protocol.minimization_tolerance_kj_mol_nm,
                protocol.minimization_steps,
            )
            simulation.minimizeEnergy(
                tolerance=protocol.minimization_tolerance_kj_mol_nm
                * kilojoule_per_mole
                / nanometer,
                maxIterations=protocol.minimization_steps,
            )
            logger.info("[%s] Minimization complete", protocol.step_name)

        if needs_velocity_initialization:
            # CHARMM-GUI draws velocities after constrained coordinates are minimized;
            # doing it before minimization can destabilize the first integration step.
            self.simulation_initializer.initialize_velocities(simulation, protocol)

        self.reporter_installer.install(
            simulation,
            artifacts,
            protocol,
            append=resumed,
        )

        logger.info(
            "[%s] Running %d remaining MD steps with dt %.4f ps",
            protocol.step_name,
            remaining_steps,
            protocol.timestep_ps,
        )
        if remaining_steps > 0:
            simulation.step(remaining_steps)
        logger.info("[%s] Dynamics complete", protocol.step_name)

        result = self.output_writer.write(simulation, artifacts, protocol)
        self.checkpoint_manager.clear(artifacts, protocol)
        return result
