import argparse
from pathlib import Path
from typing import Literal

SystemEnvironment = Literal["membrane", "solution"]

_INPUT_ROOT = Path("data/inputs/openmm_native")
_OUTPUT_ROOT = Path("data/outputs/openmm_native")


def system_id_argument(value: str) -> str:
    system_id = value.strip()
    if (
        not system_id
        or system_id in {".", ".."}
        or "/" in system_id
        or "\\" in system_id
    ):
        raise argparse.ArgumentTypeError(
            "system_id must be a single non-empty directory name"
        )
    return system_id


def system_paths(
    environment: SystemEnvironment,
    system_id: str,
) -> tuple[Path, Path]:
    return (
        _INPUT_ROOT / environment / system_id,
        _OUTPUT_ROOT / environment / system_id,
    )
