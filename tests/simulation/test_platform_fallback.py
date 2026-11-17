import os
import unittest
from unittest import mock

from charmm_gui_md.shared.simulation import OpenMMSimulationFactory


class SimulationPlatformFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("OPENMM_")
        }
        self.enterContext(mock.patch.dict(os.environ, environment, clear=True))

    def test_create_simulation_tries_gpu_platforms_before_cpu(self) -> None:
        factory = OpenMMSimulationFactory()
        cpu_platform = object()
        created_simulation = object()

        def platform_lookup(name: str):
            if name in {"HIP", "CUDA", "OpenCL"}:
                raise RuntimeError(f"{name} unavailable")
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
            mock.patch(
                "charmm_gui_md.shared.simulation.factory.Platform.getPlatformByName",
                side_effect=lookup,
            ),
            mock.patch("charmm_gui_md.shared.simulation.factory.Simulation") as ctor,
            mock.patch.object(factory, "_log_selected_platform"),
        ):
            result = factory._create_simulation(object(), object(), object(), "test")
        self.assertIs(result, ctor.return_value)
        self.assertIs(ctor.call_args.args[3], opencl)

    def test_explicit_gpu_failure_does_not_fall_back(self) -> None:
        with (
            mock.patch.dict(os.environ, {"OPENMM_PLATFORM": "OpenCL"}),
            mock.patch(
                "charmm_gui_md.shared.simulation.factory.Platform.getPlatformByName",
                side_effect=Exception("unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "OpenCL.*not available"),
        ):
            OpenMMSimulationFactory()._create_simulation(
                object(), object(), object(), "test"
            )

    def test_gpu_context_initialization_failure_tries_next_platform(self) -> None:
        created = object()

        def initialize(topology, system, integrator, platform, properties):
            if platform in {"HIP", "CUDA"}:
                raise RuntimeError(f"{platform} context failed")
            return created

        with (
            mock.patch(
                "charmm_gui_md.shared.simulation.factory.Platform.getPlatformByName",
                side_effect=lambda name: name,
            ) as lookup,
            mock.patch(
                "charmm_gui_md.shared.simulation.factory.Simulation",
                side_effect=initialize,
            ) as ctor,
            mock.patch.object(OpenMMSimulationFactory, "_log_selected_platform"),
            self.assertLogs(
                "charmm_gui_md.shared.simulation.factory", level="WARNING"
            ) as logs,
        ):
            result = OpenMMSimulationFactory()._create_simulation(
                object(), object(), object(), "stage"
            )

        self.assertIs(result, created)
        self.assertEqual(
            lookup.call_args_list,
            [mock.call("HIP"), mock.call("CUDA"), mock.call("OpenCL")],
        )
        self.assertEqual(
            [call.args[3] for call in ctor.call_args_list], ["HIP", "CUDA", "OpenCL"]
        )
        self.assertEqual(len(logs.output), 2)

    def test_reference_is_used_when_all_other_platforms_are_unavailable(self) -> None:
        reference = object()

        def lookup(name):
            if name == "Reference":
                return reference
            raise RuntimeError(f"{name} unavailable")

        with (
            mock.patch(
                "charmm_gui_md.shared.simulation.factory.Platform.getPlatformByName",
                side_effect=lookup,
            ) as get_platform,
            mock.patch("charmm_gui_md.shared.simulation.factory.Simulation") as ctor,
            mock.patch.object(OpenMMSimulationFactory, "_log_selected_platform"),
        ):
            result = OpenMMSimulationFactory()._create_simulation(
                object(), object(), object(), "stage"
            )

        self.assertIs(result, ctor.return_value)
        self.assertEqual(
            get_platform.call_args_list,
            [mock.call(name) for name in ("HIP", "CUDA", "OpenCL", "CPU", "Reference")],
        )
        self.assertIs(ctor.call_args.args[3], reference)
        self.assertEqual(ctor.call_args.args[4], {})

    def test_explicit_gpu_context_failure_does_not_fall_back(self) -> None:
        failure = RuntimeError("CUDA context failed")
        with (
            mock.patch.dict(os.environ, {"OPENMM_PLATFORM": "CUDA"}),
            mock.patch(
                "charmm_gui_md.shared.simulation.factory.Platform.getPlatformByName"
            ) as lookup,
            mock.patch(
                "charmm_gui_md.shared.simulation.factory.Simulation",
                side_effect=failure,
            ) as ctor,
            self.assertRaisesRegex(RuntimeError, "CUDA.*failed to initialize") as error,
        ):
            OpenMMSimulationFactory()._create_simulation(
                object(), object(), object(), "stage"
            )

        self.assertIs(error.exception.__cause__, failure)
        lookup.assert_called_once_with("CUDA")
        ctor.assert_called_once()

    def test_all_context_failures_report_every_attempt(self) -> None:
        with (
            mock.patch(
                "charmm_gui_md.shared.simulation.factory.Platform.getPlatformByName",
                side_effect=lambda name: name,
            ),
            mock.patch(
                "charmm_gui_md.shared.simulation.factory.Simulation",
                side_effect=RuntimeError("context failed"),
            ),
            self.assertLogs("charmm_gui_md.shared.simulation.factory", level="WARNING"),
            self.assertRaisesRegex(
                RuntimeError, "Could not initialize any OpenMM platform"
            ) as error,
        ):
            OpenMMSimulationFactory()._create_simulation(
                object(), object(), object(), "stage"
            )

        for name in ("HIP", "CUDA", "OpenCL", "CPU", "Reference"):
            self.assertIn(
                f"{name} initialization failed: context failed", str(error.exception)
            )

    def test_platform_selection_forwards_only_applicable_gpu_properties(self) -> None:
        for name, properties in (
            ("HIP", {"Precision": "mixed", "DeviceIndex": "1,2"}),
            ("CUDA", {"Precision": "mixed", "DeviceIndex": "1,2"}),
            (
                "OpenCL",
                {
                    "Precision": "mixed",
                    "DeviceIndex": "1,2",
                    "OpenCLPlatformIndex": "3",
                },
            ),
            ("CPU", {}),
            ("Reference", {}),
        ):
            with (
                self.subTest(platform=name),
                mock.patch.dict(
                    os.environ,
                    {
                        "OPENMM_PLATFORM": f" {name} ",
                        "OPENMM_DEVICE_INDEX": " 1,2 ",
                        "OPENMM_OPENCL_PLATFORM_INDEX": " 3 ",
                        "CUDA_VISIBLE_DEVICES": "GPU-test-allocation",
                    },
                ),
                mock.patch(
                    "charmm_gui_md.shared.simulation.factory.Platform.getPlatformByName"
                ) as lookup,
                mock.patch(
                    "charmm_gui_md.shared.simulation.factory.Simulation"
                ) as ctor,
                mock.patch.object(OpenMMSimulationFactory, "_log_selected_platform"),
            ):
                result = OpenMMSimulationFactory()._create_simulation(
                    object(), object(), object(), "stage"
                )
                self.assertEqual(
                    os.environ["CUDA_VISIBLE_DEVICES"], "GPU-test-allocation"
                )

            self.assertIs(result, ctor.return_value)
            lookup.assert_called_once_with(name)
            self.assertEqual(ctor.call_args.args[4], properties)

    def test_blank_device_settings_use_default_gpu_properties(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {
                    "OPENMM_PLATFORM": "OpenCL",
                    "OPENMM_DEVICE_INDEX": " ",
                    "OPENMM_OPENCL_PLATFORM_INDEX": "",
                },
            ),
            mock.patch(
                "charmm_gui_md.shared.simulation.factory.Platform.getPlatformByName"
            ),
            mock.patch("charmm_gui_md.shared.simulation.factory.Simulation") as ctor,
            mock.patch.object(OpenMMSimulationFactory, "_log_selected_platform"),
        ):
            OpenMMSimulationFactory()._create_simulation(
                object(), object(), object(), "stage"
            )

        self.assertEqual(ctor.call_args.args[4], {"Precision": "mixed"})


if __name__ == "__main__":
    unittest.main()
