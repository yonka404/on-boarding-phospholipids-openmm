from pathlib import Path

from membrane_openmm.pipeline import run_case


def main() -> None:
    inputs_root = Path("inputs/n100/charmm-gui")
    outdir = Path("results/n100")
    temperature_k = 303.15
    platform_name = "CPU"  # or None
    report_interval = 5_000
    checkpoint_interval = 50_000
    continue_from_existing = True

    run_case(
        inputs_root=inputs_root,
        outdir=outdir,
        temperature_k=temperature_k,
        platform_name=platform_name,
        report_interval=report_interval,
        checkpoint_interval=checkpoint_interval,
        continue_from_existing=continue_from_existing,
    )


if __name__ == "__main__":
    main()
