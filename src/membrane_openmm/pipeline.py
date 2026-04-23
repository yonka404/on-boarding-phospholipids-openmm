from pathlib import Path

from membrane_openmm.protocol import DEFAULT_PROTOCOL_SCHEDULE
from membrane_openmm.workflow import StageRunner, SweepRunner

PROTOCOL_STAGES = DEFAULT_PROTOCOL_SCHEDULE.stage_names

_default_stage_runner = StageRunner(protocol_schedule=DEFAULT_PROTOCOL_SCHEDULE)
_default_sweep_runner = SweepRunner(
    stage_runner=_default_stage_runner,
    protocol_schedule=DEFAULT_PROTOCOL_SCHEDULE,
)


def run_single_step(
    inputs_dir: Path,
    outputs_dir: Path,
    step_name: str,
) -> Path:
    return _default_stage_runner.run(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        step_name=step_name,
    )


def run_protocol_sweep(
    inputs_dir: Path,
    outputs_dir: Path,
) -> None:
    _default_sweep_runner.run(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
    )
