from dataclasses import dataclass
from pathlib import Path


_BOOLEAN_BY_TOKEN = {
    "yes": True,
    "no": False,
    "true": True,
    "false": False,
}


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


@dataclass(frozen=True)
class OpenMMStageProtocol:
    step_name: str
    minimization_steps: int | None
    minimization_tolerance_kj_mol_nm: float | None
    generate_velocities: bool
    velocity_temperature_kelvin: float | None
    dynamics_steps: int
    timestep_ps: float
    state_report_interval_steps: int
    trajectory_report_interval_steps: int
    temperature_kelvin: float
    friction_per_ps: float
    switch_distance_nm: float
    cutoff_distance_nm: float
    ewald_tolerance: float
    constraints_name: str | None
    pressure_coupling: bool
    pressure_bar: float | None
    barostat_kind: str | None
    membrane_xy_mode_name: str | None
    membrane_z_mode_name: str | None
    surface_tension_dyne_per_cm: float | None
    barostat_interval_steps: int | None

    @classmethod
    def from_file(cls, step_name: str, protocol_path: Path) -> "OpenMMStageProtocol":
        values = _read_key_value_file(protocol_path)
        pressure_coupling = _read_bool(values, "pcouple", default=False)

        return cls(
            step_name=step_name,
            minimization_steps=_read_optional_int(values, "mini_nstep"),
            minimization_tolerance_kj_mol_nm=_read_optional_float(values, "mini_tol"),
            generate_velocities=_read_bool(values, "gen_vel", default=False),
            velocity_temperature_kelvin=_read_optional_float(values, "gen_temp"),
            dynamics_steps=_read_required_int(values, "nstep"),
            timestep_ps=_read_required_float(values, "dt"),
            state_report_interval_steps=_read_required_int(values, "nstout"),
            trajectory_report_interval_steps=_read_required_int(values, "nstdcd"),
            temperature_kelvin=_read_required_float(values, "temp"),
            friction_per_ps=_read_required_float(values, "fric_coeff"),
            switch_distance_nm=_read_required_float(values, "r_on"),
            cutoff_distance_nm=_read_required_float(values, "r_off"),
            ewald_tolerance=_read_required_float(values, "ewald_tol"),
            constraints_name=_read_optional_str(values, "cons"),
            pressure_coupling=pressure_coupling,
            pressure_bar=_read_required_float(values, "p_ref")
            if pressure_coupling
            else None,
            barostat_kind=_read_optional_str(values, "p_type")
            if pressure_coupling
            else None,
            membrane_xy_mode_name=_read_optional_str(values, "p_xymode")
            if pressure_coupling
            else None,
            membrane_z_mode_name=_read_optional_str(values, "p_zmode")
            if pressure_coupling
            else None,
            surface_tension_dyne_per_cm=_read_optional_float(values, "p_tens")
            if pressure_coupling
            else None,
            barostat_interval_steps=_read_required_int(values, "p_freq")
            if pressure_coupling
            else None,
        )

    @property
    def has_minimization(self) -> bool:
        return (
            self.minimization_steps is not None
            and self.minimization_tolerance_kj_mol_nm is not None
        )


def _read_key_value_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Stage protocol not found: {path}")

    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line or "=" not in line:
            continue

        key, value = (part.strip() for part in line.split("=", maxsplit=1))
        values[key.lower()] = value

    return values


def _read_bool(values: dict[str, str], key: str, default: bool | None = None) -> bool:
    raw_value = values.get(key)
    if raw_value is None:
        if default is None:
            raise ValueError(f"Missing boolean protocol field: {key}")
        return default

    normalized = raw_value.strip().lower()
    if normalized not in _BOOLEAN_BY_TOKEN:
        raise ValueError(f"Unsupported boolean token for {key}: {raw_value!r}")

    return _BOOLEAN_BY_TOKEN[normalized]


def _read_required_int(values: dict[str, str], key: str) -> int:
    raw_value = values.get(key)
    if raw_value is None:
        raise ValueError(f"Missing integer protocol field: {key}")
    return int(raw_value)


def _read_optional_int(values: dict[str, str], key: str) -> int | None:
    raw_value = values.get(key)
    return None if raw_value is None else int(raw_value)


def _read_required_float(values: dict[str, str], key: str) -> float:
    raw_value = values.get(key)
    if raw_value is None:
        raise ValueError(f"Missing float protocol field: {key}")
    return float(raw_value)


def _read_optional_float(values: dict[str, str], key: str) -> float | None:
    raw_value = values.get(key)
    return None if raw_value is None else float(raw_value)


def _read_optional_str(values: dict[str, str], key: str) -> str | None:
    raw_value = values.get(key)
    if raw_value is None:
        return None

    cleaned = raw_value.strip()
    return cleaned if cleaned else None
