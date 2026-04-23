import logging
from pathlib import Path

from membrane_openmm.pipeline import run_single_step


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_single_step(
        inputs_dir=Path("data/inputs/charmmgui"),
        outputs_dir=Path("data/outputs"),
        step_name="step6.1_equilibration",
    )


if __name__ == "__main__":
    main()
