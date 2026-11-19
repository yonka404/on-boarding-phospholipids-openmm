"""Run a membrane or solution protocol directly, or select one stage with --stage."""

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mains._common import PROJECT_ROOT, system_id_argument, system_paths
from mains._console import configure_logging
from mains._gpu_monitor import monitor_gpu

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=("membrane", "solution"))
    parser.add_argument("system_id", type=system_id_argument)
    parser.add_argument("--stage", help="Run one protocol stage instead of the full protocol.")
    args = parser.parse_args(argv)

    configure_logging()
    # Keep help available without importing OpenMM or starting runtime bootstrap.
    from charmm_gui_md import membrane, solution

    workflow = membrane if args.environment == "membrane" else solution
    if args.stage is not None and args.stage not in workflow.PROTOCOL_STAGES:
        parser.error(
            f"invalid stage for {args.environment}: {args.stage!r}; "
            f"choose from {', '.join(workflow.PROTOCOL_STAGES)}"
        )

    inputs_dir, outputs_dir = system_paths(args.environment, args.system_id)
    logger.info(
        "OpenMM · %s/%s · %s · PID %d", args.environment, args.system_id,
        "full protocol" if args.stage is None else "single stage", os.getpid(),
    )
    with monitor_gpu(PROJECT_ROOT / "mains/logs", args.environment, args.system_id):
        if args.stage is None:
            workflow.run_protocol_sweep(inputs_dir=inputs_dir, outputs_dir=outputs_dir)
        else:
            workflow.run_single_step(
                inputs_dir=inputs_dir,
                outputs_dir=outputs_dir,
                step_name=args.stage,
            )


if __name__ == "__main__":
    main()
