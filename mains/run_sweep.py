import logging
from pathlib import Path

from protein_membrane_md.pipeline import run_protocol_sweep


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_protocol_sweep(
        inputs_dir=Path("data/inputs/openmm_native"),
        outputs_dir=Path("data/outputs/openmm_native"),
    )


if __name__ == "__main__":
    main()
