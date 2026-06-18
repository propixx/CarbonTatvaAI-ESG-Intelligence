#!/usr/bin/env python3
"""Prepare starter data assets for the ESG Intelligence & Benchmarking Engine.

This script uses the annual-report-derived datasets and PRD/BRSR master CSVs
already present in the repo. It does not require GPUs or model inference.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "benchmark_engine"

SOURCE_CANDIDATES = [
    ROOT / "data" / "prd_drive_download" / "Annual Reports 22-23" / "esg_kpis_22-23.csv",
    ROOT / "data" / "prd_drive_download" / "Annual Reports 22-23" / "esg_paragraph_intents_22-23.csv",
    ROOT / "data" / "drive_download" / "esg_prd_master_dataset_22-23.csv",
    ROOT / "data" / "drive_download" / "esg_prd_master_dataset_24_25.csv",
    ROOT / "data" / "drive_download" / "esg_prd_master_dataset_25-26.csv",
    ROOT / "data" / "kpi_summary" / "kpi_summary_all.jsonl",
]

DISCLOSURE_COLUMNS = [
    "has_environmental",
    "has_social",
    "has_governance",
    "has_climate_risk",
    "has_net_zero",
    "has_energy",
    "has_water",
    "has_waste",
    "has_scope_1",
    "has_scope_2",
    "has_scope_3",
    "has_diversity",
    "has_human_rights",
    "has_csr",
    "has_supply_chain",
    "has_board_governance",
    "has_tcfd",
    "has_ifrs_s1_s2",
    "has_cdp",
]

KPI_COLUMNS = [
    "kpi_scope1_emissions_tco2e_current",
    "kpi_scope2_emissions_tco2e_current",
    "kpi_scope3_emissions_tco2e_current",
    "kpi_scope1_scope2_total_tco2e_current",
    "kpi_renewable_energy_percent",
    "kpi_renewable_energy_consumption_gj",
    "kpi_total_energy_consumption_gj",
    "kpi_water_consumption_kl_current",
    "kpi_water_withdrawal_kl_current",
    "kpi_total_waste_generated_current",
    "kpi_waste_recycled_current",
    "kpi_waste_recycled_percent",
    "kpi_female_employee_percent",
    "kpi_women_on_board_percent",
    "kpi_energy_intensity_current",
    "kpi_net_zero_target_year",
]

METADATA_COLUMNS = [
    "company",
    "reporting_year",
    "meta_sector",
    "meta_market_cap",
    "meta_framework_used",
    "meta_brsr_version",
    "meta_assurance_type",
    "meta_geography",
    "top_sections",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    value = str(value).replace("\ufeff", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_company(value: Any) -> str:
    text = clean_text(value).upper()
    text = re.sub(r"\.(PDF|CSV|XLSX?|JSONL?)$", "", text)
    text = re.sub(r"[_-]\d{8,}.*$", "", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_year(value: Any, fallback: str = "") -> str:
    text = clean_text(value).lower()
    if "2025-26" in text or "25-26" in text:
        return "FY 2025-26"
    if "2024-25" in text or "24_25" in text or "24-25" in text:
        return "FY 2024-25"
    if "2022-23" in text or "22-23" in text:
        return "FY 2022-23"
    return fallback


def source_type(path: Path) -> str:
    name = path.name.lower()
    if "paragraph_intents" in name:
        return "annual_report_paragraph_intents"
    if "esg_kpis" in name:
        return "annual_report_kpis"
    if "prd_master" in name:
        return "brsr_prd_master"
    if "kpi_summary" in name:
        return "kpi_summary_dataset"
    return "unknown"


def infer_year_from_path(path: Path) -> str:
    return normalize_year(str(path))


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table: {path}")


def valid_sources() -> list[Path]:
    return [
        path
        for path in SOURCE_CANDIDATES
        if path.exists() and path.is_file() and ".part" not in path.name.lower()
    ]


def build_inventory(sources: list[Path]) -> pd.DataFrame:
    rows = []
    for path in sources:
        row: dict[str, Any] = {
            "source_path": str(path.relative_to(ROOT)),
            "source_type": source_type(path),
            "reporting_year_hint": infer_year_from_path(path),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
            "status": "ok",
            "rows": None,
            "columns": None,
        }
        try:
            if path.suffix.lower() == ".csv":
                sample = pd.read_csv(path, nrows=5)
                row["columns"] = len(sample.columns)
                row["column_names_preview"] = ", ".join(sample.columns[:12])
            elif path.suffix.lower() == ".jsonl":
                sample = pd.read_json(path, lines=True, nrows=5)
                row["columns"] = len(sample.columns)
                row["column_names_preview"] = ", ".join(sample.columns[:12])
        except Exception as exc:
            row["status"] = f"read_error: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def kpi_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    text = clean_text(value)
    return text not in {"", "nan", "NaN", "None", "null"}


def build_company_year_table(master_paths: list[Path], annual_kpi_paths: list[Path]) -> pd.DataFrame:
    frames = []

    for path in master_paths:
        df = pd.read_csv(path)
        df["source_type"] = "brsr_prd_master"
        df["source_file"] = str(path.relative_to(ROOT))
        df["reporting_year"] = df["reporting_year"].map(
            lambda value: normalize_year(value, infer_year_from_path(path))
        )
        frames.append(df)

    for path in annual_kpi_paths:
        df = pd.read_csv(path)
        rename = {
            "scope1_emissions_tco2e_current": "kpi_scope1_emissions_tco2e_current",
            "scope2_emissions_tco2e_current": "kpi_scope2_emissions_tco2e_current",
            "scope3_emissions_tco2e_current": "kpi_scope3_emissions_tco2e_current",
            "renewable_energy_percent": "kpi_renewable_energy_percent",
            "renewable_energy_consumption_gj": "kpi_renewable_energy_consumption_gj",
            "total_energy_consumption_gj": "kpi_total_energy_consumption_gj",
            "water_consumption_kl_current": "kpi_water_consumption_kl_current",
            "water_withdrawal_kl_current": "kpi_water_withdrawal_kl_current",
            "total_waste_generated_current": "kpi_total_waste_generated_current",
            "waste_recycled_current": "kpi_waste_recycled_current",
            "waste_recycled_percent": "kpi_waste_recycled_percent",
            "female_employee_percent": "kpi_female_employee_percent",
            "women_on_board_percent": "kpi_women_on_board_percent",
            "energy_intensity_current": "kpi_energy_intensity_current",
            "net_zero_target_year": "kpi_net_zero_target_year",
        }
        df = df.rename(columns=rename)
        df["source_type"] = "annual_report_kpis"
        df["source_file"] = str(path.relative_to(ROOT))
        df["reporting_year"] = df.get("reporting_year", "").map(
            lambda value: normalize_year(value, infer_year_from_path(path))
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["company_normalized"] = combined["company"].map(normalize_company)
    combined["sector"] = combined.get("meta_sector", "").map(clean_text)
    combined["sector"] = combined["sector"].replace("", "Unknown")

    for column in DISCLOSURE_COLUMNS:
        if column not in combined:
            combined[column] = False
        combined[column] = combined[column].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
            if not isinstance(value, bool)
            else value
        )
    for column in KPI_COLUMNS:
        if column not in combined:
            combined[column] = pd.NA

    combined["disclosure_coverage_count"] = combined[DISCLOSURE_COLUMNS].sum(axis=1)
    combined["kpi_available_count"] = combined[KPI_COLUMNS].apply(
        lambda row: sum(kpi_present(value) for value in row),
        axis=1,
    )
    combined["benchmark_quality_score"] = (
        combined["disclosure_coverage_count"] * 2 + combined["kpi_available_count"]
    )

    output_columns = [
        "company",
        "company_normalized",
        "reporting_year",
        "sector",
        "source_type",
        "source_file",
        "meta_market_cap",
        "meta_framework_used",
        "meta_brsr_version",
        "meta_assurance_type",
        "top_sections",
        "disclosure_coverage_count",
        "kpi_available_count",
        "benchmark_quality_score",
        *DISCLOSURE_COLUMNS,
        *KPI_COLUMNS,
    ]
    for column in output_columns:
        if column not in combined:
            combined[column] = pd.NA

    combined = combined[output_columns].sort_values(
        ["company_normalized", "reporting_year", "benchmark_quality_score"],
        ascending=[True, True, False],
    )
    combined = combined.drop_duplicates(
        subset=["company_normalized", "reporting_year", "source_type"],
        keep="first",
    ).reset_index(drop=True)
    return combined


def build_adoption(company_year: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if company_year.empty:
        return pd.DataFrame()
    grouped = company_year.groupby(["reporting_year", "sector"], dropna=False)
    for (year, sector), group in grouped:
        total = len(group)
        for column in DISCLOSURE_COLUMNS:
            count = int(group[column].fillna(False).astype(bool).sum())
            rows.append(
                {
                    "reporting_year": year,
                    "sector": sector,
                    "metric_type": "disclosure",
                    "metric": column.removeprefix("has_"),
                    "companies_total": total,
                    "companies_disclosing": count,
                    "adoption_rate_percent": round((count / total) * 100, 2) if total else 0,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["reporting_year", "sector", "metric_type", "metric"]
    )


def build_kpi_availability(company_year: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if company_year.empty:
        return pd.DataFrame()
    grouped = company_year.groupby(["reporting_year", "sector"], dropna=False)
    for (year, sector), group in grouped:
        total = len(group)
        for column in KPI_COLUMNS:
            count = int(group[column].map(kpi_present).sum())
            rows.append(
                {
                    "reporting_year": year,
                    "sector": sector,
                    "metric_type": "kpi",
                    "metric": column.removeprefix("kpi_"),
                    "companies_total": total,
                    "companies_with_kpi": count,
                    "availability_rate_percent": round((count / total) * 100, 2) if total else 0,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["reporting_year", "sector", "metric_type", "metric"]
    )


def summarize_annual_intents(path: Path, output_dir: Path, chunksize: int = 50_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    intent_counter: dict[tuple[str, str, str], int] = defaultdict(int)
    section_counter: dict[tuple[str, str, str], int] = defaultdict(int)
    samples = []
    per_intent_samples = Counter()
    total_rows = 0

    usecols = [
        "company",
        "reporting_year",
        "primary_intent",
        "section_labels",
        "paragraph_text",
        "page",
    ]
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=lambda col: col in usecols):
        total_rows += len(chunk)
        chunk["company_normalized"] = chunk["company"].map(normalize_company)
        chunk["reporting_year"] = chunk.get("reporting_year", "").map(
            lambda value: normalize_year(value, infer_year_from_path(path))
        )
        for row in chunk.itertuples(index=False):
            company = getattr(row, "company_normalized", "")
            year = getattr(row, "reporting_year", infer_year_from_path(path))
            intent = clean_text(getattr(row, "primary_intent", "")) or "unclassified"
            intent_counter[(company, year, intent)] += 1

            labels = clean_text(getattr(row, "section_labels", ""))
            if labels:
                for label in [part.strip() for part in labels.split(",") if part.strip()]:
                    section_counter[(company, year, label)] += 1

            if per_intent_samples[intent] < 5:
                samples.append(
                    {
                        "company": clean_text(getattr(row, "company", "")),
                        "company_normalized": company,
                        "reporting_year": year,
                        "page": getattr(row, "page", None),
                        "primary_intent": intent,
                        "section_labels": labels,
                        "paragraph_text": clean_text(getattr(row, "paragraph_text", ""))[:1000],
                    }
                )
                per_intent_samples[intent] += 1

    intent_rows = [
        {
            "company_normalized": company,
            "reporting_year": year,
            "primary_intent": intent,
            "paragraph_count": count,
        }
        for (company, year, intent), count in intent_counter.items()
    ]
    section_rows = [
        {
            "company_normalized": company,
            "reporting_year": year,
            "section_label": section,
            "paragraph_count": count,
        }
        for (company, year, section), count in section_counter.items()
    ]

    intent_df = pd.DataFrame(intent_rows)
    section_df = pd.DataFrame(section_rows)
    samples_df = pd.DataFrame(samples)
    intent_df.to_csv(output_dir / "annual_report_intent_counts.csv", index=False)
    section_df.to_csv(output_dir / "annual_report_section_counts.csv", index=False)
    samples_df.to_csv(output_dir / "annual_report_intent_samples.csv", index=False)

    metadata = {
        "source": str(path.relative_to(ROOT)),
        "total_paragraph_rows_processed": total_rows,
        "sample_rows": len(samples_df),
    }
    (output_dir / "annual_report_intent_processing.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return intent_df, samples_df


def write_manifest(output_dir: Path, inventory: pd.DataFrame, company_year: pd.DataFrame) -> None:
    manifest = {
        "product": "ESG Intelligence & Benchmarking Engine",
        "tagline": "Benchmark your disclosures against industry leaders, identify gaps, and strengthen ESG reporting before publication.",
        "created_assets": [
            "source_inventory.csv",
            "benchmark_company_year.csv",
            "disclosure_adoption_by_sector_year.csv",
            "kpi_availability_by_sector_year.csv",
            "annual_report_intent_counts.csv",
            "annual_report_section_counts.csv",
            "annual_report_intent_samples.csv",
        ],
        "source_files": inventory.to_dict("records"),
        "company_year_rows": int(len(company_year)),
        "unique_companies": int(company_year["company_normalized"].nunique()) if not company_year.empty else 0,
        "years": sorted(company_year["reporting_year"].dropna().unique().tolist()) if not company_year.empty else [],
        "sectors": sorted(company_year["sector"].dropna().unique().tolist()) if not company_year.empty else [],
        "notes": [
            "Incomplete .part downloads are intentionally excluded.",
            "Annual Reports 22-23 are represented by extracted KPI and paragraph intent datasets available locally.",
            "The next step is dashboard/API logic for peer selection and gap analysis.",
        ],
    }
    (output_dir / "benchmark_engine_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-heavy-intents", action="store_true")
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = valid_sources()
    inventory = build_inventory(sources)
    inventory.to_csv(output_dir / "source_inventory.csv", index=False)

    master_paths = [path for path in sources if source_type(path) == "brsr_prd_master"]
    annual_kpi_paths = [path for path in sources if source_type(path) == "annual_report_kpis"]
    company_year = build_company_year_table(master_paths, annual_kpi_paths)
    company_year.to_csv(output_dir / "benchmark_company_year.csv", index=False)

    adoption = build_adoption(company_year)
    adoption.to_csv(output_dir / "disclosure_adoption_by_sector_year.csv", index=False)

    kpi_availability = build_kpi_availability(company_year)
    kpi_availability.to_csv(output_dir / "kpi_availability_by_sector_year.csv", index=False)

    if not args.skip_heavy_intents:
        intent_paths = [path for path in sources if source_type(path) == "annual_report_paragraph_intents"]
        for path in intent_paths:
            summarize_annual_intents(path, output_dir)

    write_manifest(output_dir, inventory, company_year)

    print(f"Benchmark engine starter data written to: {output_dir}")
    print(f"Sources inventoried: {len(inventory)}")
    print(f"Company-year rows: {len(company_year)}")
    print(f"Unique companies: {company_year['company_normalized'].nunique() if not company_year.empty else 0}")
    print("Created:")
    for path in sorted(output_dir.iterdir()):
        print(f"- {path.name} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
