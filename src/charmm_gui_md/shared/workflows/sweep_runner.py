import logging
from pathlib import Path

from charmm_gui_md.shared.artifacts import StageArtifacts
from charmm_gui_md.shared.profile import SystemProfile
from charmm_gui_md.shared.workflows.stage_runner import StageRunner

logger = logging.getLogger(__name__)


class SweepRunner:
    def __init__(
        self,
        profile: SystemProfile,
        stage_runner: StageRunner | None = None,
    ) -> None:
        self.profile = profile
        self.stage_runner = stage_runner or StageRunner(profile=profile)

    def run(
        self,
        inputs_dir: Path,
        outputs_dir: Path,
    ) -> None:
        # Inspect the whole sequence before a new stage can create any output.
        incomplete_stage = None
        for step_name in self.profile.protocol_schedule.stage_names:
            artifacts = StageArtifacts.for_stage(outputs_dir, step_name)
            if incomplete_stage is not None and artifacts.output_dir.exists():
                raise RuntimeError(
                    f"Inconsistent stage order: {step_name} exists after incomplete "
                    f"{incomplete_stage}. Inspect outputs before retrying."
                )
            artifacts.validate_for_run()
            if not artifacts.is_complete or artifacts.checkpoint_path.is_file():
                incomplete_stage = incomplete_stage or step_name

        for step_name in self.profile.protocol_schedule.stage_names:
            artifacts = StageArtifacts.for_stage(outputs_dir, step_name)
            if artifacts.is_complete and not artifacts.checkpoint_path.is_file():
                logger.info(
                    "Skipping completed stage %s; final state and coordinates exist",
                    step_name,
                )
                continue

            logger.info("Running %s", step_name)
            self.stage_runner.run(
                inputs_dir=inputs_dir,
                outputs_dir=outputs_dir,
                step_name=step_name,
            )
