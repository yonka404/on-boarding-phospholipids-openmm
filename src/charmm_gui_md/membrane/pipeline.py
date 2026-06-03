from pathlib import Path

from charmm_gui_md.membrane.profile import MEMBRANE_PROFILE
from charmm_gui_md.shared.workflows import StageRunner, SweepRunner

PROTOCOL_STAGES = MEMBRANE_PROFILE.protocol_schedule.stage_names


def run_single_step(
    inputs_dir: Path,
    outputs_dir: Path,
    step_name: str,
) -> Path:
    return StageRunner(profile=MEMBRANE_PROFILE).run(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        step_name=step_name,
    )


def run_protocol_sweep(
    inputs_dir: Path,
    outputs_dir: Path,
) -> None:
    SweepRunner(profile=MEMBRANE_PROFILE).run(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
    )
