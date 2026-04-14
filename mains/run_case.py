import argparse
from pathlib import Path

from membrane_openmm.pipeline import run_case


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run one CHARMM-GUI membrane case with OpenMM"
    )
    parser.add_argument(
        "--system-root", required=True, help="Path to the CHARMM-GUI folder"
    )
    parser.add_argument("--outdir", required=True, help="Output directory for this run")
    parser.add_argument(
        "--temperature", type=float, default=303.15, help="Simulation temperature in K"
    )
    parser.add_argument(
        "--platform", default=None, help="OpenMM platform: CPU, CUDA, HIP, OpenCL"
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=5000,
        help="CSV/DCD report interval in steps",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50000,
        help="Checkpoint interval in steps",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Do not skip already-finished stages",
    )

    args = parser.parse_args(argv)

    run_case(
        outdir=Path(args.outdir),
        temperature_k=args.temperature,
        platform_name=args.platform,
        report_interval=args.report_interval,
        checkpoint_interval=args.checkpoint_interval,
        continue_from_existing=not args.fresh,
    )


if __name__ == "__main__":
    main(
        [
            "--system-root",
            "inputs/n100",
            "--outdir",
            "results/n100",
            "--platform",
            "CPU",
            "--temperature",
            "303.15",
        ]
    )
