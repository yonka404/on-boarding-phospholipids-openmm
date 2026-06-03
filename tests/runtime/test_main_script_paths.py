import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


class MainScriptPathTests(unittest.TestCase):
    def test_membrane_single_step_uses_system_id_and_stage_arguments(self) -> None:
        module = _load_main("run_membrane_single_step.py")

        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "run_membrane_single_step.py",
                    "ligand_membrane",
                    "step6.3_equilibration",
                ],
            ),
            mock.patch.object(module, "run_single_step") as run_single_step,
        ):
            module.main()

        run_single_step.assert_called_once_with(
            inputs_dir=Path("data/inputs/openmm_native/membrane/ligand_membrane"),
            outputs_dir=Path("data/outputs/openmm_native/membrane/ligand_membrane"),
            step_name="step6.3_equilibration",
        )

    def test_membrane_sweep_uses_system_id_argument(self) -> None:
        module = _load_main("run_membrane_sweep.py")

        with (
            mock.patch.object(
                sys,
                "argv",
                ["run_membrane_sweep.py", "ligand_membrane"],
            ),
            mock.patch.object(module, "run_protocol_sweep") as run_protocol_sweep,
        ):
            module.main()

        run_protocol_sweep.assert_called_once_with(
            inputs_dir=Path("data/inputs/openmm_native/membrane/ligand_membrane"),
            outputs_dir=Path("data/outputs/openmm_native/membrane/ligand_membrane"),
        )

    def test_solution_single_step_uses_system_id_and_stage_arguments(self) -> None:
        module = _load_main("run_solution_single_step.py")

        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "run_solution_single_step.py",
                    "abeta_40",
                    "step5_production",
                ],
            ),
            mock.patch.object(module, "run_single_step") as run_single_step,
        ):
            module.main()

        run_single_step.assert_called_once_with(
            inputs_dir=Path("data/inputs/openmm_native/solution/abeta_40"),
            outputs_dir=Path("data/outputs/openmm_native/solution/abeta_40"),
            step_name="step5_production",
        )

    def test_solution_sweep_uses_system_id_argument(self) -> None:
        module = _load_main("run_solution_sweep.py")

        with (
            mock.patch.object(sys, "argv", ["run_solution_sweep.py", "abeta_40"]),
            mock.patch.object(module, "run_protocol_sweep") as run_protocol_sweep,
        ):
            module.main()

        run_protocol_sweep.assert_called_once_with(
            inputs_dir=Path("data/inputs/openmm_native/solution/abeta_40"),
            outputs_dir=Path("data/outputs/openmm_native/solution/abeta_40"),
        )

    def test_scripts_reject_system_ids_with_path_separators(self) -> None:
        module = _load_main("run_solution_sweep.py")

        with (
            mock.patch.object(sys, "argv", ["run_solution_sweep.py", "../abeta_40"]),
            self.assertRaises(SystemExit),
        ):
            module.main()


def _load_main(filename: str):
    path = Path("mains") / filename
    spec = importlib.util.spec_from_file_location(f"test_main_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
