#!/usr/bin/env python3
"""Run a starter ESG benchmark gap analysis from prepared benchmark data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from prepare_benchmark_engine_data import (
    DISCLOSURE_COLUMNS,
    KPI_COLUMNS,
    ROOT,
    clean_text,
    kpi_present,
    normalize_company,
)


DATA_DIR = ROOT / "artifacts" / "benchmark_engine"


def pct(value: float) -> str:
    return f"{value:.2f}%"


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    company_year = pd.read_csv(data_dir / "benchmark_company_year.csv")
    adoption = pd.read_csv(data_dir / "disclosure_adoption_by_sector_year.csv")
    kpi_availability = pd.read_csv(data_dir / "kpi_availability_by_sector_year.csv")
    return company_year, adoption, kpi_availability


def pick_target(company_year: pd.DataFrame, company: str | None, year: str | None) -> pd.Series:
    if company:
        normalized = normalize_company(company)
        matches = company_year[company_year["company_normalized"].str.contains(normalized, na=False)]
        if year:
            matches = matches[matches["reporting_year"].eq(year)]
        if matches.empty:
            raise ValueError(f"No company-year row found for company={company!r}, year={year!r}")
        return matches.sort_values("benchmark_quality_score", ascending=False).iloc[0]

    return company_year.sort_values("benchmark_quality_score", ascending=False).iloc[0]


def peer_examples(
    company_year: pd.DataFrame,
    year: str,
    sector: str,
    column: str,
    limit: int = 5,
) -> list[str]:
    peers = company_year[
        company_year["reporting_year"].eq(year)
        & company_year["sector"].eq(sector)
        & company_year[column].map(lambda value: bool(value) if column.startswith("has_") else kpi_present(value))
    ]
    names = peers["company"].dropna().astype(str).drop_duplicates().head(limit).tolist()
    return names


def benchmark(
    company_year: pd.DataFrame,
    adoption: pd.DataFrame,
    kpi_availability: pd.DataFrame,
    target: pd.Series,
    adoption_threshold: float,
) -> dict[str, Any]:
    year = clean_text(target["reporting_year"])
    sector = clean_text(target["sector"]) or "Unknown"

    disclosure_stats = adoption[
        adoption["reporting_year"].eq(year)
        & adoption["sector"].eq(sector)
        & adoption["adoption_rate_percent"].ge(adoption_threshold)
    ]
    kpi_stats = kpi_availability[
        kpi_availability["reporting_year"].eq(year)
        & kpi_availability["sector"].eq(sector)
        & kpi_availability["availability_rate_percent"].ge(adoption_threshold)
    ]

    missing_disclosures = []
    for row in disclosure_stats.itertuples(index=False):
        column = f"has_{row.metric}"
        if column in target and not bool(target[column]):
            missing_disclosures.append(
                {
                    "finding": f"{row.metric.replace('_', ' ').title()} disclosure is missing.",
                    "sector_adoption": pct(float(row.adoption_rate_percent)),
                    "evidence": f"{int(row.companies_disclosing)} of {int(row.companies_total)} peer rows disclose this in {sector} for {year}.",
                    "peer_examples": peer_examples(company_year, year, sector, column),
                    "suggested_addition": f"Add a {row.metric.replace('_', ' ')} disclosure with evidence and KPIs where available.",
                    "priority": "High" if row.adoption_rate_percent >= 75 else "Medium",
                }
            )

    kpi_gaps = []
    for row in kpi_stats.itertuples(index=False):
        column = f"kpi_{row.metric}"
        if column in target and not kpi_present(target[column]):
            kpi_gaps.append(
                {
                    "finding": f"{row.metric.replace('_', ' ').title()} KPI is missing.",
                    "sector_availability": pct(float(row.availability_rate_percent)),
                    "evidence": f"{int(row.companies_with_kpi)} of {int(row.companies_total)} peer rows provide this KPI in {sector} for {year}.",
                    "peer_examples": peer_examples(company_year, year, sector, column),
                    "suggested_addition": f"Start tracking and disclosing {row.metric.replace('_', ' ')}.",
                    "priority": "High" if row.availability_rate_percent >= 75 else "Medium",
                }
            )

    covered_disclosures = [
        column.removeprefix("has_")
        for column in DISCLOSURE_COLUMNS
        if column in target and bool(target[column])
    ]
    available_kpis = [
        column.removeprefix("kpi_")
        for column in KPI_COLUMNS
        if column in target and kpi_present(target[column])
    ]

    return {
        "target": {
            "company": clean_text(target["company"]),
            "company_normalized": clean_text(target["company_normalized"]),
            "reporting_year": year,
            "sector": sector,
            "source_type": clean_text(target["source_type"]),
            "source_file": clean_text(target["source_file"]),
            "disclosure_coverage_count": int(target["disclosure_coverage_count"]),
            "kpi_available_count": int(target["kpi_available_count"]),
        },
        "executive_summary": {
            "missing_disclosures": len(missing_disclosures),
            "kpi_gaps": len(kpi_gaps),
            "covered_disclosures": covered_disclosures,
            "available_kpis": available_kpis,
        },
        "missing_disclosures": missing_disclosures,
        "kpi_gaps": kpi_gaps,
        "recommendations": [
            item["suggested_addition"]
            for item in [*missing_disclosures[:5], *kpi_gaps[:5]]
        ],
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    target = result["target"]
    lines = [
        f"# Benchmark Report - {target['company']}",
        "",
        f"- Year: {target['reporting_year']}",
        f"- Sector: {target['sector']}",
        f"- Source: {target['source_file']}",
        f"- Missing disclosures: {result['executive_summary']['missing_disclosures']}",
        f"- KPI gaps: {result['executive_summary']['kpi_gaps']}",
        "",
        "## Missing Disclosures",
    ]
    for item in result["missing_disclosures"]:
        lines.extend(
            [
                f"- **{item['finding']}**",
                f"  Evidence: {item['evidence']}",
                f"  Peer examples: {', '.join(item['peer_examples']) or 'Not available'}",
                f"  Suggested addition: {item['suggested_addition']}",
            ]
        )
    if not result["missing_disclosures"]:
        lines.append("- No high-adoption disclosure gaps detected at the selected threshold.")

    lines.extend(["", "## KPI Gaps"])
    for item in result["kpi_gaps"]:
        lines.extend(
            [
                f"- **{item['finding']}**",
                f"  Evidence: {item['evidence']}",
                f"  Peer examples: {', '.join(item['peer_examples']) or 'Not available'}",
                f"  Suggested addition: {item['suggested_addition']}",
            ]
        )
    if not result["kpi_gaps"]:
        lines.append("- No high-adoption KPI gaps detected at the selected threshold.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--company")
    parser.add_argument("--year")
    parser.add_argument("--threshold", type=float, default=60.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    company_year, adoption, kpi_availability = load_data(args.data_dir)
    target = pick_target(company_year, args.company, args.year)
    result = benchmark(company_year, adoption, kpi_availability, target, args.threshold)

    output_json = args.output_json or args.data_dir / "sample_benchmark_report.json"
    output_md = args.output_md or args.data_dir / "sample_benchmark_report.md"
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, output_md)

    print(json.dumps(result["target"], indent=2, ensure_ascii=False))
    print("Missing disclosures:", len(result["missing_disclosures"]))
    print("KPI gaps:", len(result["kpi_gaps"]))
    print("Wrote:", output_json)
    print("Wrote:", output_md)


if __name__ == "__main__":
    main()
