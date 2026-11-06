"""Print one read-only snapshot of this checkout's simulations, locally or on SIMCALC."""

from __future__ import annotations

import sys

# Set this before importing any project modules: status must not create caches.
sys.dont_write_bytecode = True

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mains._common import PROJECT_ROOT
from mains._console import color_enabled
from mains._status import collect_snapshot, render_snapshot


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        snapshot = collect_snapshot(PROJECT_ROOT)
        sys.stdout.write(render_snapshot(
            snapshot, width=shutil.get_terminal_size((120, 24)).columns,
            color=color_enabled(sys.stdout),
        ))
    except KeyboardInterrupt:
        print("Status interrupted; simulations continue.", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
