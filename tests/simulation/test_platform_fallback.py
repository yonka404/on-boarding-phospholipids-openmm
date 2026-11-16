import os

os.environ["CHARMM_GUI_MD_OPENMM_SKIP_BOOTSTRAP"] = "1"

import unittest
from unittest import mock

from charmm_gui_md.shared.simulation import OpenMMSimulationFactory


class SimulationPlatformFallbackTests(unittest.TestCase):
    def test_create_simulation_tries_gpu_platforms_before_cpu(self) -> None:
        factory = OpenMMSimulationFactory()
        cpu_platform = object()
        created_simulation = object()

        def platform_lookup(name: str):
            if name in {"HIP", "CUDA", "OpenCL"}:
                raise Exception(f"{name} unavailable")
            if name == "CPU":
                return cpu_platform
            raise AssertionError(name)

        with (
            mock.patch(
                "charmm_gui_md.shared.simulation.factory.Platform.getPlatformByName",
                side_effect=platform_lookup,
            ) as get_platform,
            mock.patch(
                "charmm_gui_md.shared.simulation.factory.Simulation",
                return_value=created_simulation,
            ) as simulation_ctor,
            mock.patch.object(factory, "_log_selected_platform"),
        ):
            simulation = factory._create_simulation(
                topology=object(),
                system=object(),
                integrator=object(),
                step_name="step6.1_equilibration",
            )

        self.assertIs(simulation, created_simulation)
        self.assertEqual(
            [call.args[0] for call in get_platform.call_args_list],
            ["HIP", "CUDA", "OpenCL", "CPU"],
        )
        self.assertIs(simulation_ctor.call_args.args[3], cpu_platform)

    def test_missing_hip_and_cuda_selects_opencl(self) -> None:
        factory = OpenMMSimulationFactory()
        opencl = object()

        def lookup(name):
            if name in {"HIP", "CUDA"}:
                raise RuntimeError("unavailable")
            if name == "OpenCL":
                return opencl
            raise AssertionError("CPU must not be selected when OpenCL works")

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("charmm_gui_md.shared.simulation.factory.Platform.getPlatformByName", side_effect=lookup),
            mock.patch("charmm_gui_md.shared.simulation.factory.Simulation") as ctor,
            mock.patch.object(factory, "_log_selected_platform"),
        ):
            result = factory._create_simulation(object(), object(), object(), "test")
        self.assertIs(result, ctor.return_value)
        self.assertIs(ctor.call_args.args[3], opencl)

    def test_explicit_gpu_failure_does_not_fall_back(self) -> None:
        with (
            mock.patch.dict(os.environ, {"OPENMM_PLATFORM": "OpenCL"}),
            mock.patch("charmm_gui_md.shared.simulation.factory.Platform.getPlatformByName", side_effect=Exception("unavailable")),
            self.assertRaisesRegex(RuntimeError, "OpenCL.*not available"),
        ):
            OpenMMSimulationFactory()._create_simulation(object(), object(), object(), "test")


if __name__ == "__main__":
    unittest.main()
