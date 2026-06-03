from dataclasses import dataclass
from pathlib import Path

_BOOLEAN_BY_TOKEN = {
    "yes": True,
    "no": False,
    "true": True,
    "false": False,
}


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
    coulomb_method_name: str = "PME"
    vdw_method_name: str = "Force-switch"
    restraints_enabled: bool = False
    protein_backbone_restraint_kj_mol_nm2: float = 0.0
    protein_side_chain_restraint_kj_mol_nm2: float = 0.0
    lipid_position_restraint_kj_mol_nm2: float = 0.0
    lipid_dihedral_restraint_kj_mol_rad2: float = 0.0
    carbohydrate_dihedral_restraint_kj_mol_rad2: float = 0.0

    @classmethod
    def from_file(cls, step_name: str, protocol_path: Path) -> "OpenMMStageProtocol":
        values = _ProtocolValues(_read_protocol_values(protocol_path))
        pressure_coupling = values.bool("pcouple", default=False)

        return cls(
            step_name=step_name,
            minimization_steps=values.optional_int("mini_nstep"),
            minimization_tolerance_kj_mol_nm=values.optional_float("mini_tol"),
            generate_velocities=values.bool("gen_vel", default=False),
            velocity_temperature_kelvin=values.optional_float("gen_temp"),
            dynamics_steps=values.required_int("nstep"),
            timestep_ps=values.required_float("dt"),
            state_report_interval_steps=values.required_int("nstout"),
            trajectory_report_interval_steps=values.required_int("nstdcd"),
            temperature_kelvin=values.required_float("temp"),
            friction_per_ps=values.required_float("fric_coeff"),
            switch_distance_nm=values.required_float("r_on"),
            cutoff_distance_nm=values.required_float("r_off"),
            ewald_tolerance=values.required_float("ewald_tol"),
            constraints_name=values.optional_str("cons"),
            pressure_coupling=pressure_coupling,
            pressure_bar=values.required_float("p_ref") if pressure_coupling else None,
            barostat_kind=values.optional_str("p_type") if pressure_coupling else None,
            membrane_xy_mode_name=values.optional_str("p_xymode")
            if pressure_coupling
            else None,
            membrane_z_mode_name=values.optional_str("p_zmode")
            if pressure_coupling
            else None,
            surface_tension_dyne_per_cm=values.optional_float("p_tens")
            if pressure_coupling
            else None,
            barostat_interval_steps=values.required_int("p_freq")
            if pressure_coupling
            else None,
            coulomb_method_name=values.string("coulomb", default="PME"),
            vdw_method_name=values.string("vdw", default="Force-switch"),
            restraints_enabled=values.bool("rest", default=False),
            protein_backbone_restraint_kj_mol_nm2=values.float_or_default(
                "fc_bb",
                default=0.0,
            ),
            protein_side_chain_restraint_kj_mol_nm2=values.float_or_default(
                "fc_sc",
                default=0.0,
            ),
            lipid_position_restraint_kj_mol_nm2=values.float_or_default(
                "fc_lpos",
                default=0.0,
            ),
            lipid_dihedral_restraint_kj_mol_rad2=values.float_or_default(
                "fc_ldih",
                default=0.0,
            ),
            carbohydrate_dihedral_restraint_kj_mol_rad2=values.float_or_default(
                "fc_cdih",
                default=0.0,
            ),
        )

    @property
    def has_minimization(self) -> bool:
        return (
            self.minimization_steps is not None
            and self.minimization_tolerance_kj_mol_nm is not None
        )


def _read_protocol_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Stage protocol not found: {path}")

    return _read_assignment_protocol_values(path.read_text())


class _ProtocolValues:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def string(self, key: str, *, default: str | None = None) -> str:
        raw_value = self.values.get(key)
        if raw_value is None:
            if default is None:
                raise ValueError(f"Missing string protocol field: {key}")
            return default

        return raw_value.strip()

    def optional_str(self, key: str) -> str | None:
        raw_value = self.values.get(key)
        if raw_value is None:
            return None

        cleaned = raw_value.strip()
        return cleaned if cleaned else None

    def required_int(self, key: str) -> int:
        raw_value = self.values.get(key)
        if raw_value is None:
            raise ValueError(f"Missing integer protocol field: {key}")
        return int(raw_value)

    def optional_int(self, key: str) -> int | None:
        raw_value = self.values.get(key)
        return None if raw_value is None else int(raw_value)

    def required_float(self, key: str) -> float:
        raw_value = self.values.get(key)
        if raw_value is None:
            raise ValueError(f"Missing float protocol field: {key}")
        return float(raw_value)

    def optional_float(self, key: str) -> float | None:
        raw_value = self.values.get(key)
        return None if raw_value is None else float(raw_value)

    def float_or_default(self, key: str, *, default: float | None = None) -> float:
        raw_value = self.values.get(key)
        if raw_value is None:
            if default is None:
                raise ValueError(f"Missing float protocol field: {key}")
            return default

        return float(raw_value)

    def bool(self, key: str, *, default: bool | None = None) -> bool:
        raw_value = self.values.get(key)
        if raw_value is None:
            if default is None:
                raise ValueError(f"Missing boolean protocol field: {key}")
            return default

        normalized = raw_value.strip().lower()
        if normalized not in _BOOLEAN_BY_TOKEN:
            raise ValueError(f"Unsupported boolean token for {key}: {raw_value!r}")

        return _BOOLEAN_BY_TOKEN[normalized]


def _read_assignment_protocol_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = _strip_inline_comment(raw_line).strip()
        if not line or "=" not in line:
            continue

        key, value = (part.strip() for part in line.split("=", maxsplit=1))
        if any(char.isspace() for char in key):
            continue

        values[key.lower()] = value

    return values


def _strip_inline_comment(line: str) -> str:
    return line.split("#", maxsplit=1)[0].split("!", maxsplit=1)[0]
