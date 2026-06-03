import argparse
import logging

from charmm_gui_md.shared.cli import system_id_argument, system_paths
from charmm_gui_md.solution import run_protocol_sweep


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run a solution MD protocol sweep.")
    parser.add_argument("system_id", type=system_id_argument)
    args = parser.parse_args()

    inputs_dir, outputs_dir = system_paths("solution", args.system_id)
    run_protocol_sweep(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
    )


if __name__ == "__main__":
    main()
