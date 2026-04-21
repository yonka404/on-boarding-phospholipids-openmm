from pathlib import Path

from membrane_openmm.pipeline import run_single_step

STEP_NAMES = (
    "step6.1_equilibration",
    "step6.2_equilibration",
    "step6.3_equilibration",
    "step6.4_equilibration",
    "step6.5_equilibration",
    "step6.6_equilibration",
    "step7_production",
)


def main() -> None:
    inputs_dir = Path("data/inputs/charmmgui")
    outputs_dir = Path("data/outputs")

    for step_name in STEP_NAMES:
        print(f"=== Running {step_name} ===")
        run_single_step(
            inputs_dir=inputs_dir,
            outputs_dir=outputs_dir,
            step_name=step_name,
        )


if __name__ == "__main__":
    main()