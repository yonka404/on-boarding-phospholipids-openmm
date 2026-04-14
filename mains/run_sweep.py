#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from membrane_openmm.pipeline import run_case


def infer_case_name(system_root: Path) -> str:
    if system_root.name == "charmm-gui" and system_root.parent.name:
        return system_root.parent.name
    return system_root.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a size sweep across multiple CHARMM-GUI inputs"
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True, help="List of CHARMM-GUI system folders"
    )
    parser.add_argument(
        "--results-root", required=True, help="Root folder for output runs"
    )
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
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    for system in args.systems:
        system_root = Path(system).expanduser().resolve()
        case_name = infer_case_name(system_root)
        outdir = results_root / case_name
        print(f"\n=== Running {case_name} ===")
        run_case(
            system_root=system_root,
            outdir=outdir,
            temperature_k=args.temperature,
            platform_name=args.platform,
            report_interval=args.report_interval,
            checkpoint_interval=args.checkpoint_interval,
            continue_from_existing=True,
        )


if __name__ == "__main__":
    main()
