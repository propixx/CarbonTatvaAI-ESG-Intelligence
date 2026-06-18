#!/usr/bin/env python3
"""Convert multiple ESG KPI CSV files into one KPI-to-report dataset."""

import argparse
import csv
import json
from pathlib import Path

from convert_kpi_to_report import make_example


def find_csvs(paths: list[str]) -> list[Path]:
    csvs: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix.lower() == ".csv":
            csvs.append(path)
        elif path.is_dir():
            csvs.extend(sorted(path.rglob("*.csv")))
    return sorted(set(csvs))


def load_examples(csv_paths: list[Path]) -> tuple[list[dict[str, str]], dict[str, int]]:
    examples: list[dict[str, str]] = []
    manifest: dict[str, int] = {}
    seen: set[tuple[str, str, str]] = set()

    for csv_path in csv_paths:
        count = 0
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                example = make_example(row)
                if not example:
                    continue
                key = (example["instruction"], example["input"], example["output"])
                if key in seen:
                    continue
                seen.add(key)
                examples.append(example)
                count += 1
        manifest[str(csv_path)] = count
        print(f"{csv_path}: {count} examples")

    return examples, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="CSV files or directories containing CSV files.")
    parser.add_argument("--output", default="data/kpi_to_esg_report_all_years.json")
    parser.add_argument("--manifest", default="data/kpi_to_esg_report_all_years_manifest.json")
    args = parser.parse_args()

    csv_paths = find_csvs(args.paths)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in: {args.paths}")

    examples, manifest = load_examples(csv_paths)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(examples, handle, indent=2, ensure_ascii=False)

    manifest_path = Path(args.manifest)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"total_examples": len(examples), "sources": manifest}, handle, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(examples)} examples to {output_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
