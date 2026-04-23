from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from membrane_openmm.protocol import ProtocolSchedule


FINAL_COORDINATES_FILENAME = "final_coordinates.pdb"
FINAL_STATE_FILENAME = "final_state.xml"
STATE_DATA_FILENAME = "state_data.csv"
TRAJECTORY_FILENAME = "trajectory.dcd"


@dataclass(frozen=True)
class StageArtifacts:
    step_name: str
    output_dir: Path

    @classmethod
    def for_stage(cls, output_root: Path, step_name: str) -> "StageArtifacts":
        return cls(
            step_name=step_name,
            output_dir=output_root / step_name,
        )

    def create_directory(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def final_coordinates_path(self) -> Path:
        return self.output_dir / FINAL_COORDINATES_FILENAME

    @property
    def final_state_path(self) -> Path:
        return self.output_dir / FINAL_STATE_FILENAME

    @property
    def state_data_path(self) -> Path:
        return self.output_dir / STATE_DATA_FILENAME

    @property
    def trajectory_path(self) -> Path:
        return self.output_dir / TRAJECTORY_FILENAME


@dataclass(frozen=True)
class RestartSource:
    coordinates_path: Path
    state_path: Path | None
    description: str


class RestartResolver:
    def __init__(self, protocol_schedule: ProtocolSchedule) -> None:
        self._protocol_schedule = protocol_schedule

    def resolve(
        self,
        inputs_dir: Path,
        outputs_dir: Path,
        step_name: str,
    ) -> RestartSource:
        previous_stage = self._protocol_schedule.previous_stage(step_name)
        if previous_stage is None:
            return RestartSource(
                coordinates_path=inputs_dir / "step5_assembly.pdb",
                state_path=None,
                description="initial CHARMM-GUI coordinates",
            )

        previous_artifacts = StageArtifacts.for_stage(outputs_dir, previous_stage)
        previous_state = previous_artifacts.final_state_path

        return RestartSource(
            coordinates_path=previous_artifacts.final_coordinates_path,
            state_path=previous_state if previous_state.is_file() else None,
            description=f"restart from previous protocol stage '{previous_stage}'",
        )
