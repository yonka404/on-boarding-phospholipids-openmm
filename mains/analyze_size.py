#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def summarize_case(case_dir: Path, tail_fraction: float) -> dict[str, float | int | str]:
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    prod_dir = None
    for candidate in sorted(case_dir.iterdir()):
        if candidate.is_dir() and candidate.name.endswith("_prod"):
            prod_dir = candidate
            break
    if prod_dir is None:
        raise FileNotFoundError(f"Could not find production folder inside {case_dir}")

    thermo_csv = next(prod_dir.glob("*.thermo.csv"))
    rows = read_csv_rows(thermo_csv)
    if not rows:
        raise ValueError(f"No rows found in {thermo_csv}")

    start = int(len(rows) * (1.0 - tail_fraction))
    tail = rows[start:]

    def f(key: str) -> float:
        return mean(float(r[key]) for r in tail)

    nliptop = int(metadata["metadata"]["nliptop"])
    nlipbot = int(metadata["metadata"]["nlipbot"])

    return {
        "case": case_dir.name,
        "temperature_k": metadata["temperature_k"],
        "nliptop": nliptop,
        "nlipbot": nlipbot,
        "total_lipids": nliptop + nlipbot,
        "mean_temperature_k": round(f("temperature_k"), 6),
        "mean_potential_energy_kj_mol": round(f("potential_energy_kj_mol"), 6),
        "mean_area_xy_nm2": round(f("area_xy_nm2"), 6),
        "mean_area_per_lipid_nm2": round(f("area_per_lipid_nm2"), 6),
        "mean_lz_nm": round(f("lz_nm"), 6),
        "mean_volume_nm3": round(f("volume_nm3"), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize size-effect metrics from finished runs")
    parser.add_argument("--results-root", required=True, help="Folder containing one subfolder per case")
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.5,
        help="Fraction of the production trajectory tail to average (default: 0.5)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    rows: list[dict[str, float | int | str]] = []
    for case_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        try:
            rows.append(summarize_case(case_dir, args.tail_fraction))
        except Exception as exc:
            print(f"[skip] {case_dir.name}: {exc}")

    if not rows:
        raise SystemExit("No completed cases found")

    out_csv = results_root / "size_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_csv}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
