import logging
from pathlib import Path

from protein_membrane_md.protocols import DEFAULT_PROTOCOL_SCHEDULE, ProtocolSchedule
from protein_membrane_md.workflows.stage_runner import StageRunner

logger = logging.getLogger(__name__)


class SweepRunner:
    def __init__(
        self,
        stage_runner: StageRunner | None = None,
        protocol_schedule: ProtocolSchedule | None = None,
    ) -> None:
        self.protocol_schedule = protocol_schedule or DEFAULT_PROTOCOL_SCHEDULE
        self.stage_runner = stage_runner or StageRunner(
            protocol_schedule=self.protocol_schedule
        )

    def run(
        self,
        inputs_dir: Path,
        outputs_dir: Path,
    ) -> None:
        for step_name in self.protocol_schedule.stage_names:
            logger.info("Running %s", step_name)
            self.stage_runner.run(
                inputs_dir=inputs_dir,
                outputs_dir=outputs_dir,
                step_name=step_name,
            )
