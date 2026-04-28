from dataclasses import dataclass
from pathlib import Path
import re

_BOOLEAN_BY_TOKEN = {
    "yes": True,
    "no": False,
    "true": True,
    "false": False,
}

_REQUIRED_PROTOCOL_KEYS = (
    "nstep",
    "dt",
    "nstout",
    "nstdcd",
    "temp",
    "fric_coeff",
    "r_on",
    "r_off",
    "ewald_tol",
)
_CHARMM_SET_RE = re.compile(r"^\s*set\s+([A-Za-z0-9_]+)\s*=\s*([^\s!]+)", re.I)
_CHARMM_MINIMIZATION_RE = re.compile(r"\bmini\s+\w+\s+nstep\s+(\d+)", re.I)


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

    text = path.read_text()
    values = _read_assignment_protocol_values(text)
    if all(key in values for key in _REQUIRED_PROTOCOL_KEYS):
        return values

    charmm_values = _read_charmm_gui_protocol_values(text)
    if charmm_values:
        return {**values, **charmm_values}

    return values


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


def _read_charmm_gui_protocol_values(text: str) -> dict[str, str]:
    tokens = _charmm_tokens(text)
    lowered_tokens = [token.lower() for token in tokens]
    if "dyna" not in lowered_tokens:
        return {}

    set_values = _charmm_set_values(text)
    values: dict[str, str] = {}

    nstep = set_values.get("nstep") or _token_after(lowered_tokens, tokens, "nstep")
    if nstep is not None:
        values["nstep"] = _resolve_charmm_value(nstep, set_values)

    timestep = _token_after(lowered_tokens, tokens, "timestp", "timestep")
    if timestep is not None:
        values["dt"] = _resolve_charmm_value(timestep, set_values)

    nstout = (
        _token_after(lowered_tokens, tokens, "nprint")
        or _token_after(lowered_tokens, tokens, "isvfrq")
        or "1000"
    )
    values["nstout"] = _resolve_charmm_value(nstout, set_values)

    nstdcd = _token_after(lowered_tokens, tokens, "nsavc")
    if nstdcd is None or int(_resolve_charmm_value(nstdcd, set_values)) <= 0:
        nstdcd = _token_after(lowered_tokens, tokens, "ntrfrq") or nstout
    values["nstdcd"] = _resolve_charmm_value(nstdcd, set_values)

    temperature = (
        set_values.get("temp")
        or _token_after(lowered_tokens, tokens, "finalt")
        or _token_after(lowered_tokens, tokens, "firstt")
    )
    if temperature is not None:
        values["temp"] = _resolve_charmm_value(temperature, set_values)

    gamma = _token_after(lowered_tokens, tokens, "gamma") or "1"
    values["fric_coeff"] = _resolve_charmm_value(gamma, set_values)

    ctonnb = _token_after(lowered_tokens, tokens, "ctonnb")
    if ctonnb is not None:
        values["r_on"] = _angstrom_to_nanometer(
            _resolve_charmm_value(ctonnb, set_values)
        )

    ctofnb = _token_after(lowered_tokens, tokens, "ctofnb")
    if ctofnb is not None:
        values["r_off"] = _angstrom_to_nanometer(
            _resolve_charmm_value(ctofnb, set_values)
        )

    if "ewald" in lowered_tokens:
        values["ewald_tol"] = "0.0005"

    if "shake" in lowered_tokens and "bonh" in lowered_tokens:
        values["cons"] = "HBonds"

    generate_velocities = "start" in lowered_tokens and "restart" not in lowered_tokens
    values["gen_vel"] = "yes" if generate_velocities else "no"
    if generate_velocities and "temp" in values:
        values["gen_temp"] = values["temp"]

    minimization_steps = sum(
        int(match.group(1)) for match in _CHARMM_MINIMIZATION_RE.finditer(text)
    )
    if minimization_steps:
        values["mini_nstep"] = str(minimization_steps)
        values["mini_tol"] = "100.0"

    pressure_coupling = "cpt" in lowered_tokens or "prmc" in lowered_tokens
    values["pcouple"] = "yes" if pressure_coupling else "no"
    if pressure_coupling:
        values["p_ref"] = _resolve_charmm_value(
            _token_after(lowered_tokens, tokens, "przz") or "1.0",
            set_values,
        )
        values["p_freq"] = _resolve_charmm_value(
            _token_after(lowered_tokens, tokens, "iprsfrq") or "15",
            set_values,
        )
        values["p_type"] = "membrane" if "prmc" in lowered_tokens else "isotropic"
        values["p_xymode"] = "XYIsotropic"
        values["p_zmode"] = "ZFree"
        values["p_tens"] = _resolve_charmm_value(
            _token_after(lowered_tokens, tokens, "tens") or "0.0",
            set_values,
        )

    return values


def _charmm_set_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        match = _CHARMM_SET_RE.match(raw_line)
        if match is not None:
            values[match.group(1).lower()] = match.group(2)

    return values


def _charmm_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_line in text.splitlines():
        line = _strip_inline_comment(raw_line).strip()
        if not line or line.startswith("*"):
            continue

        tokens.extend(line.rstrip("-").split())

    return tokens


def _strip_inline_comment(line: str) -> str:
    return line.split("#", maxsplit=1)[0].split("!", maxsplit=1)[0]


def _token_after(
    lowered_tokens: list[str],
    tokens: list[str],
    *names: str,
) -> str | None:
    for name in names:
        try:
            token_index = lowered_tokens.index(name)
        except ValueError:
            continue

        value_index = token_index + 1
        if value_index < len(tokens):
            return tokens[value_index]

    return None


def _resolve_charmm_value(value: str, set_values: dict[str, str]) -> str:
    cleaned = value.strip().rstrip(",")
    if cleaned.startswith("@"):
        return set_values[cleaned[1:].lower()]

    return cleaned


def _angstrom_to_nanometer(value: str) -> str:
    return f"{float(value) / 10.0:g}"


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
