from pathlib import Path

from membrane_openmm.pipeline import run_case


def main() -> None:
    # Remember to set the working directory path correctly in your IDE!
    run_case(
        inputs_dir=Path("data/inputs/charmm-gui"), outputs_dir=Path("data/outputs")
    )


if __name__ == "__main__":
    main()
