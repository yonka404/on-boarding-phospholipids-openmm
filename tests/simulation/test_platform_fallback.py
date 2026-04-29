import os

os.environ["MEMBRANE_OPENMM_SKIP_BOOTSTRAP"] = "1"

import unittest
from unittest import mock

from protein_membrane_md.simulation import OpenMMSimulationFactory


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
                "protein_membrane_md.simulation.factory.Platform.getPlatformByName",
                side_effect=platform_lookup,
            ) as get_platform,
            mock.patch(
                "protein_membrane_md.simulation.factory.Simulation",
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


if __name__ == "__main__":
    unittest.main()
