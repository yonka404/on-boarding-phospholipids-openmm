"""Launcher arguments and paths. Keep this module compatible with Python 3.8."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

SystemEnvironment = Literal["membrane", "solution"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    *,
    project_root: Path | None = None,
) -> tuple[Path, Path]:
    root = PROJECT_ROOT if project_root is None else project_root.resolve()
    return (
        root / "data/inputs/openmm_native" / environment / system_id,
        root / "data/outputs/openmm_native" / environment / system_id,
    )
