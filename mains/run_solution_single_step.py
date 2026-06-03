import argparse
import logging

from charmm_gui_md.shared.cli import system_id_argument, system_paths
from charmm_gui_md.solution import PROTOCOL_STAGES, run_single_step


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run one solution MD protocol stage.")
    parser.add_argument("system_id", type=system_id_argument)
    parser.add_argument("step_name", choices=PROTOCOL_STAGES)
    args = parser.parse_args()

    inputs_dir, outputs_dir = system_paths("solution", args.system_id)
    run_single_step(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        step_name=args.step_name,
    )


if __name__ == "__main__":
    main()
