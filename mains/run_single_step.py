from pathlib import Path

from membrane_openmm.pipeline import run_single_step


def main() -> None:
    run_single_step(
        inputs_dir=Path("data/inputs/charmmgui"),
        outputs_dir=Path("data/outputs"),
        step_name="step6.1_equilibration",
        starting_pdb=None,
    )


if __name__ == "__main__":
    main()
