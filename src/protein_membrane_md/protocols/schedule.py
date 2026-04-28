from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolSchedule:
    stage_names: tuple[str, ...]

    def require_stage(self, step_name: str) -> None:
        if step_name not in self.stage_names:
            supported_steps = ", ".join(self.stage_names)
            raise ValueError(
                f"Unsupported step_name {step_name!r}. Expected one of: {supported_steps}"
            )

    def previous_stage(self, step_name: str) -> str | None:
        self.require_stage(step_name)
        step_index = self.stage_names.index(step_name)
        if step_index == 0:
            return None
        return self.stage_names[step_index - 1]


DEFAULT_PROTOCOL_SCHEDULE = ProtocolSchedule(
    stage_names=(
        "step6.1_equilibration",
        "step6.2_equilibration",
        "step6.3_equilibration",
        "step6.4_equilibration",
        "step6.5_equilibration",
        "step6.6_equilibration",
        "step7_production",
    )
)
