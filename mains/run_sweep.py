import logging
from pathlib import Path

from membrane_openmm.pipeline import PROTOCOL_STAGES, run_single_step


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    inputs_dir = Path("data/inputs/charmmgui")
    outputs_dir = Path("data/outputs")

    logger = logging.getLogger(__name__)

    for step_name in PROTOCOL_STAGES:
        logger.info("Running %s", step_name)
        run_single_step(
            inputs_dir=inputs_dir,
            outputs_dir=outputs_dir,
            step_name=step_name,
        )


if __name__ == "__main__":
    main()