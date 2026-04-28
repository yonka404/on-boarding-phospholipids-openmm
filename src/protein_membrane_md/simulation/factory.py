import logging
import os
from collections.abc import Mapping

from openmm import (
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    MonteCarloMembraneBarostat,
    Platform,
)
from openmm.app import AllBonds, HAngles, HBonds, PME, Simulation
from openmm.unit import angstrom, bar, kelvin, nanometer, picosecond

from protein_membrane_md.inputs import SimulationInputFiles
from protein_membrane_md.protocols import OpenMMStageProtocol

logger = logging.getLogger(__name__)

_CONSTRAINT_BY_NAME = {
    "hbonds": HBonds,
    "allbonds": AllBonds,
    "hangles": HAngles,
    "none": None,
}

_MEMBRANE_XY_MODE_BY_NAME = {
    "xyisotropic": "XYIsotropic",
    "xyanisotropic": "XYAnisotropic",
    "constantarea": "ConstantArea",
}

_MEMBRANE_Z_MODE_BY_NAME = {
    "zfree": "ZFree",
    "zfixed": "ZFixed",
    "constantvolume": "ConstantVolume",
}

_GPU_PLATFORM_NAMES = ("HIP", "CUDA", "OpenCL")
_PLATFORM_PREFERENCE = (*_GPU_PLATFORM_NAMES, "CPU", "Reference")
_DEVICE_NAME_KEYS = ("Name", "name", "DeviceName", "deviceName")


class OpenMMSimulationFactory:
    def create(
        self,
        files: SimulationInputFiles,
        protocol: OpenMMStageProtocol,
    ) -> Simulation:
        psf = files.psf_file
        params = files.params_file

        a_length, b_length, c_length = files.box_lengths_angstrom
        psf.setBox(a_length * angstrom, b_length * angstrom, c_length * angstrom)

        system = psf.createSystem(
            params,
            nonbondedMethod=PME,
            nonbondedCutoff=protocol.cutoff_distance_nm * nanometer,
            switchDistance=protocol.switch_distance_nm * nanometer,
            ewaldErrorTolerance=protocol.ewald_tolerance,
            constraints=self._constraint_from_name(protocol.constraints_name),
        )

        barostat = self._build_barostat(protocol)
        if barostat is not None:
            system.addForce(barostat)

        integrator = LangevinMiddleIntegrator(
            protocol.temperature_kelvin * kelvin,
            protocol.friction_per_ps / picosecond,
            protocol.timestep_ps * picosecond,
        )

        return self._create_simulation(
            topology=psf.topology,
            system=system,
            integrator=integrator,
            step_name=protocol.step_name,
        )

    def _create_simulation(
        self,
        topology,
        system,
        integrator,
        step_name: str,
    ) -> Simulation:
        platform_override = self._read_env_value("OPENMM_PLATFORM")
        candidate_names = (
            [platform_override] if platform_override else list(_PLATFORM_PREFERENCE)
        )
        failures: list[str] = []

        for platform_name in candidate_names:
            try:
                platform = Platform.getPlatformByName(platform_name)
            except Exception as exc:
                if platform_override:
                    raise RuntimeError(
                        f"[{step_name}] OPENMM_PLATFORM={platform_name!r} was requested "
                        "but is not available in this OpenMM installation."
                    ) from exc

                logger.info(
                    "[%s] Skipping unavailable OpenMM platform %s: %s",
                    step_name,
                    platform_name,
                    exc,
                )
                failures.append(f"{platform_name} unavailable: {exc}")
                continue

            properties = self._platform_properties(platform_name)

            try:
                simulation = Simulation(
                    topology,
                    system,
                    integrator,
                    platform,
                    properties,
                )
            except Exception as exc:
                if platform_override:
                    raise RuntimeError(
                        f"[{step_name}] OPENMM_PLATFORM={platform_name!r} was requested "
                        "but failed to initialize."
                    ) from exc

                logger.warning(
                    "[%s] OpenMM platform %s failed to initialize; trying next fallback: %s",
                    step_name,
                    platform_name,
                    exc,
                )
                failures.append(f"{platform_name} initialization failed: {exc}")
                continue

            self._log_selected_platform(simulation, step_name, properties)
            return simulation

        details = (
            "; ".join(failures) if failures else "no OpenMM platforms were available"
        )
        raise RuntimeError(
            f"[{step_name}] Could not initialize any OpenMM platform. {details}"
        )

    def _platform_properties(self, platform_name: str) -> dict[str, str]:
        properties: dict[str, str] = {}

        if platform_name in _GPU_PLATFORM_NAMES:
            properties["Precision"] = "mixed"

            device_index = self._read_env_value("OPENMM_DEVICE_INDEX")
            if device_index is not None:
                properties["DeviceIndex"] = device_index

            if platform_name == "OpenCL":
                opencl_platform_index = self._read_env_value(
                    "OPENMM_OPENCL_PLATFORM_INDEX"
                )
                if opencl_platform_index is not None:
                    properties["OpenCLPlatformIndex"] = opencl_platform_index

        return properties

    def _log_selected_platform(
        self,
        simulation: Simulation,
        step_name: str,
        requested_properties: dict[str, str],
    ) -> None:
        platform = simulation.context.getPlatform()
        platform_name = platform.getName()

        resolved_properties: dict[str, str] = {}
        for property_name in platform.getPropertyNames():
            if property_name not in {
                "Precision",
                "DeviceIndex",
                "OpenCLPlatformIndex",
                "UseCpuPme",
            }:
                continue

            try:
                resolved_properties[property_name] = platform.getPropertyValue(
                    simulation.context,
                    property_name,
                )
            except Exception:
                fallback_value = requested_properties.get(property_name)
                if fallback_value is not None:
                    resolved_properties[property_name] = fallback_value

        if platform_name in _GPU_PLATFORM_NAMES:
            gpu_name = self._get_selected_gpu_name(platform, resolved_properties)
            if gpu_name is not None:
                logger.info(
                    "[%s] Using OpenMM platform %s on GPU %s with properties %s",
                    step_name,
                    platform_name,
                    gpu_name,
                    resolved_properties or requested_properties,
                )
                return

        logger.info(
            "[%s] Using OpenMM platform %s with properties %s",
            step_name,
            platform_name,
            resolved_properties or requested_properties,
        )

    def _get_selected_gpu_name(
        self,
        platform: Platform,
        resolved_properties: dict[str, str],
    ) -> str | None:
        try:
            devices = list(platform.getDevices(resolved_properties))
        except Exception:
            try:
                devices = list(platform.getDevices())
            except Exception:
                return None

        if not devices:
            return None

        descriptions = [self._describe_device(device) for device in devices]
        descriptions = [description for description in descriptions if description]
        if not descriptions:
            return None

        return " + ".join(descriptions)

    def _describe_device(self, device) -> str:
        if isinstance(device, Mapping):
            name = next(
                (
                    str(device[key])
                    for key in _DEVICE_NAME_KEYS
                    if key in device and str(device[key]).strip()
                ),
                None,
            )
            index = next(
                (
                    str(device[key])
                    for key in ("DeviceIndex", "deviceIndex", "Index", "index")
                    if key in device and str(device[key]).strip()
                ),
                None,
            )

            if name and index:
                return f"{name} (device {index})"
            if name:
                return name
            return ", ".join(f"{key}={value}" for key, value in sorted(device.items()))

        return str(device)

    @staticmethod
    def _read_env_value(name: str) -> str | None:
        value = os.getenv(name)
        if value is None:
            return None

        cleaned = value.strip()
        return cleaned if cleaned else None

    def _constraint_from_name(self, constraints_name: str | None):
        if constraints_name is None:
            return None

        normalized = constraints_name.strip().lower()
        if normalized not in _CONSTRAINT_BY_NAME:
            supported = ", ".join(sorted(_CONSTRAINT_BY_NAME))
            raise ValueError(
                f"Unsupported constraints setting {constraints_name!r}. "
                f"Expected one of: {supported}"
            )

        return _CONSTRAINT_BY_NAME[normalized]

    def _build_barostat(self, protocol: OpenMMStageProtocol):
        if not protocol.pressure_coupling:
            return None

        if protocol.pressure_bar is None or protocol.barostat_interval_steps is None:
            raise ValueError(
                f"[{protocol.step_name}] Pressure coupling is enabled but the protocol "
                "is missing barostat settings."
            )

        barostat_kind = (protocol.barostat_kind or "isotropic").strip().lower()
        if barostat_kind == "membrane":
            xy_mode_name = (
                (protocol.membrane_xy_mode_name or "XYIsotropic").strip().lower()
            )
            z_mode_name = (protocol.membrane_z_mode_name or "ZFree").strip().lower()

            if xy_mode_name not in _MEMBRANE_XY_MODE_BY_NAME:
                supported_xy = ", ".join(sorted(_MEMBRANE_XY_MODE_BY_NAME))
                raise ValueError(
                    f"Unsupported membrane XY mode {protocol.membrane_xy_mode_name!r}. "
                    f"Expected one of: {supported_xy}"
                )

            if z_mode_name not in _MEMBRANE_Z_MODE_BY_NAME:
                supported_z = ", ".join(sorted(_MEMBRANE_Z_MODE_BY_NAME))
                raise ValueError(
                    f"Unsupported membrane Z mode {protocol.membrane_z_mode_name!r}. "
                    f"Expected one of: {supported_z}"
                )

            surface_tension = (
                (protocol.surface_tension_dyne_per_cm or 0.0) * 10.0 * bar * nanometer
            )

            xy_mode_attr = _MEMBRANE_XY_MODE_BY_NAME[xy_mode_name]
            try:
                xy_mode = getattr(MonteCarloMembraneBarostat, xy_mode_attr)
            except AttributeError as exc:
                raise ValueError(
                    f"OpenMM build does not expose membrane XY mode {xy_mode_attr!r}"
                ) from exc

            z_mode_attr = _MEMBRANE_Z_MODE_BY_NAME[z_mode_name]
            try:
                z_mode = getattr(MonteCarloMembraneBarostat, z_mode_attr)
            except AttributeError as exc:
                raise ValueError(
                    f"OpenMM build does not expose membrane Z mode {z_mode_attr!r}"
                ) from exc

            return MonteCarloMembraneBarostat(
                protocol.pressure_bar * bar,
                surface_tension,
                protocol.temperature_kelvin * kelvin,
                xy_mode,
                z_mode,
                protocol.barostat_interval_steps,
            )

        return MonteCarloBarostat(
            protocol.pressure_bar * bar,
            protocol.temperature_kelvin * kelvin,
            protocol.barostat_interval_steps,
        )
