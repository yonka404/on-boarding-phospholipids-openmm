import logging
from pathlib import Path

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
        for step_name in self.profile.protocol_schedule.stage_names:
            logger.info("Running %s", step_name)
            self.stage_runner.run(
                inputs_dir=inputs_dir,
                outputs_dir=outputs_dir,
                step_name=step_name,
            )
