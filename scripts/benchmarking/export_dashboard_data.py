#!/usr/bin/env python3
"""Export benchmark artifacts into a static dashboard data file."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from comparative_benchmark import DISCLOSURE_LABELS, KPI_META


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts" / "benchmark_engine" / "benchmark_company_year.csv"
REPORT_LINKS = ROOT / "artifacts" / "benchmark_engine" / "benchmark_company_year_report_links.csv"
OUTPUT = ROOT / "dashboard" / "dashboard_data.js"


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value.replace("\ufeff", " ")).strip()
        return text or None
    return value


def display_company(value: Any) -> str:
    text = clean_value(value) or "Unknown Company"
    if re.search(r"_\d{8,}", text):
        text = text.split("_")[0]
    text = text.replace(".pdf", "").replace("_", " ")
    text = re.sub(r"\bBRSR\b.*$", "", text, flags=re.IGNORECASE).strip(" -_")
    return re.sub(r"\s+", " ", text).strip() or "Unknown Company"


def main() -> None:
    df = pd.read_csv(SOURCE)
    report_columns = [
        "company_normalized",
        "reporting_year",
        "brsr_report_count",
        "annual_report_count",
        "annual_extracted_source_count",
        "report_count",
        "report_types_available",
        "example_report_path",
    ]
    if REPORT_LINKS.exists():
        links = pd.read_csv(REPORT_LINKS)
        available_report_columns = [column for column in report_columns if column in links.columns]
        if {"company_normalized", "reporting_year"}.issubset(available_report_columns):
            links = links[available_report_columns].copy()
            count_columns = [
                column
                for column in [
                    "brsr_report_count",
                    "annual_report_count",
                    "annual_extracted_source_count",
                    "report_count",
                ]
                if column in links.columns
            ]
            for column in count_columns:
                links[column] = pd.to_numeric(links[column], errors="coerce").fillna(0)
            aggregate = {column: "max" for column in count_columns}
            for column in ["report_types_available", "example_report_path"]:
                if column in links.columns:
                    aggregate[column] = lambda values: next(
                        (clean_value(value) for value in values if clean_value(value)),
                        None,
                    )
            links = links.groupby(["company_normalized", "reporting_year"], dropna=False).agg(aggregate).reset_index()
            df = df.merge(links, on=["company_normalized", "reporting_year"], how="left")

    wanted_columns = [
        "company",
        "company_normalized",
        "reporting_year",
        "sector",
        "source_type",
        "benchmark_quality_score",
        "disclosure_coverage_count",
        "kpi_available_count",
        "brsr_report_count",
        "annual_report_count",
        "annual_extracted_source_count",
        "report_count",
        "report_types_available",
        "example_report_path",
        *DISCLOSURE_LABELS.keys(),
        *KPI_META.keys(),
    ]
    columns = [column for column in wanted_columns if column in df.columns]
    records = []
    for _, row in df[columns].iterrows():
        item = {column: clean_value(row[column]) for column in columns}
        item["company_display"] = display_company(item.get("company"))
        records.append(item)

    payload = {
        "generated_from": str(SOURCE.relative_to(ROOT)),
        "row_count": len(records),
        "kpis": KPI_META,
        "disclosures": DISCLOSURE_LABELS,
        "rows": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "window.BENCHMARK_DASHBOARD_DATA = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")
    print(f"Rows: {len(records)}")


if __name__ == "__main__":
    main()
