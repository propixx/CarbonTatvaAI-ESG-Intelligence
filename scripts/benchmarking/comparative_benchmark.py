#!/usr/bin/env python3
"""Company vs sector/custom peer-group comparative ESG benchmark.

This is the V1 logic behind questions like:

    "How does Company A's Scope 1 emissions compare with cement peers?"
    "Which KPIs are missing compared with my selected peer group?"

It uses the prepared benchmark company-year table and produces JSON + Markdown
outputs that can later be connected to a dashboard.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "artifacts" / "benchmark_engine"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "benchmark_engine"

KPI_META = {
    "kpi_scope1_emissions_tco2e_current": {
        "label": "Scope 1 emissions",
        "unit": "tCO2e",
        "lower_is_better": True,
    },
    "kpi_scope2_emissions_tco2e_current": {
        "label": "Scope 2 emissions",
        "unit": "tCO2e",
        "lower_is_better": True,
    },
    "kpi_scope3_emissions_tco2e_current": {
        "label": "Scope 3 emissions",
        "unit": "tCO2e",
        "lower_is_better": True,
    },
    "kpi_scope1_scope2_total_tco2e_current": {
        "label": "Scope 1 + Scope 2 emissions",
        "unit": "tCO2e",
        "lower_is_better": True,
    },
    "kpi_renewable_energy_percent": {
        "label": "Renewable energy share",
        "unit": "%",
        "lower_is_better": False,
    },
    "kpi_total_energy_consumption_gj": {
        "label": "Total energy consumption",
        "unit": "GJ",
        "lower_is_better": True,
    },
    "kpi_water_consumption_kl_current": {
        "label": "Water consumption",
        "unit": "KL",
        "lower_is_better": True,
    },
    "kpi_water_withdrawal_kl_current": {
        "label": "Water withdrawal",
        "unit": "KL",
        "lower_is_better": True,
    },
    "kpi_total_waste_generated_current": {
        "label": "Total waste generated",
        "unit": "",
        "lower_is_better": True,
    },
    "kpi_waste_recycled_percent": {
        "label": "Waste recycled/recovered share",
        "unit": "%",
        "lower_is_better": False,
    },
    "kpi_female_employee_percent": {
        "label": "Female employee share",
        "unit": "%",
        "lower_is_better": False,
    },
    "kpi_women_on_board_percent": {
        "label": "Women on Board",
        "unit": "%",
        "lower_is_better": False,
    },
    "kpi_energy_intensity_current": {
        "label": "Energy intensity",
        "unit": "",
        "lower_is_better": True,
    },
}

DISCLOSURE_LABELS = {
    "has_environmental": "Environmental disclosure",
    "has_social": "Social disclosure",
    "has_governance": "Governance disclosure",
    "has_climate_risk": "Climate risk",
    "has_net_zero": "Net-zero target",
    "has_energy": "Energy",
    "has_water": "Water",
    "has_waste": "Waste",
    "has_scope_1": "Scope 1",
    "has_scope_2": "Scope 2",
    "has_scope_3": "Scope 3",
    "has_diversity": "Diversity",
    "has_human_rights": "Human rights",
    "has_csr": "CSR",
    "has_supply_chain": "Supply chain",
    "has_board_governance": "Board governance",
    "has_tcfd": "TCFD",
    "has_ifrs_s1_s2": "IFRS S1/S2",
    "has_cdp": "CDP",
}


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\ufeff", " ")).strip()


def normalize_company(value: Any) -> str:
    text = clean_text(value).upper()
    text = re.sub(r"\b(LIMITED|LTD|PRIVATE|PVT|INDIA|CO|COMPANY)\b", " ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def numeric(value: Any) -> float | None:
    text = clean_text(value)
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return None
    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group(0))
    if math.isnan(number):
        return None
    return number


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    return text in {"true", "1", "yes", "y"} or (text not in {"", "false", "0", "no", "nan"})


def load_company_year(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "benchmark_company_year.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark table: {path}")
    df = pd.read_csv(path)
    df["company_normalized_runtime"] = df["company"].map(normalize_company)
    return df


def pick_target(df: pd.DataFrame, company: str, year: str | None) -> pd.Series:
    key = normalize_company(company)
    matches = df[
        df["company_normalized_runtime"].str.contains(re.escape(key), na=False)
        | df["company"].astype(str).str.contains(company, case=False, na=False, regex=False)
    ]
    if year:
        matches = matches[matches["reporting_year"].eq(year)]
    if matches.empty:
        raise ValueError(f"No benchmark row found for company={company!r}, year={year!r}")
    return matches.sort_values("benchmark_quality_score", ascending=False).iloc[0]


def select_peers(
    df: pd.DataFrame,
    target: pd.Series,
    sector: str | None,
    peers: str | None,
    year: str | None,
) -> tuple[pd.DataFrame, str]:
    target_year = year or clean_text(target["reporting_year"])
    candidates = df[df["reporting_year"].eq(target_year)].copy()

    if peers:
        keys = [normalize_company(item) for item in peers.split(",") if item.strip()]
        mask = pd.Series(False, index=candidates.index)
        for key in keys:
            mask = mask | candidates["company_normalized_runtime"].str.contains(re.escape(key), na=False)
        peer_df = candidates[mask]
        label = "custom peer group"
    else:
        sector_name = sector or clean_text(target["sector"]) or "Unknown"
        peer_df = candidates[candidates["sector"].eq(sector_name)]
        label = f"{sector_name} sector"

    target_key = clean_text(target.get("company_normalized_runtime")) or normalize_company(target["company"])
    peer_df = peer_df[peer_df["company_normalized_runtime"].ne(target_key)]
    peer_df = peer_df.drop_duplicates(subset=["company_normalized_runtime", "reporting_year"])
    return peer_df, label


def format_value(value: float | None, unit: str = "") -> str:
    if value is None:
        return "Not disclosed"
    if abs(value) >= 1000:
        text = f"{value:,.2f}".rstrip("0").rstrip(".")
    else:
        text = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{text} {unit}".strip()


def compare_kpi(target_value: float, peer_values: list[float], lower_is_better: bool) -> dict[str, Any]:
    series = pd.Series(peer_values, dtype="float64")
    mean = float(series.mean())
    median = float(series.median())
    min_value = float(series.min())
    max_value = float(series.max())

    all_values = peer_values + [target_value]
    if lower_is_better:
        rank = 1 + sum(1 for value in peer_values if value < target_value)
        better_or_equal = sum(1 for value in peer_values if target_value <= value)
        median_signal = "better than or equal to peer median" if target_value <= median else "worse than peer median"
    else:
        rank = 1 + sum(1 for value in peer_values if value > target_value)
        better_or_equal = sum(1 for value in peer_values if target_value >= value)
        median_signal = "better than or equal to peer median" if target_value >= median else "worse than peer median"

    percentile = 100.0 * better_or_equal / len(peer_values) if peer_values else 0.0
    return {
        "peer_count": len(peer_values),
        "peer_average": mean,
        "peer_median": median,
        "peer_min": min_value,
        "peer_max": max_value,
        "rank_among_group_including_company": rank,
        "group_size_including_company": len(all_values),
        "percentile_vs_peers": percentile,
        "interpretation": median_signal,
    }


def build_report(
    df: pd.DataFrame,
    target: pd.Series,
    peer_df: pd.DataFrame,
    peer_group_label: str,
    adoption_threshold: float,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    missing_kpis: list[dict[str, Any]] = []

    for column, meta in KPI_META.items():
        if column not in df.columns:
            continue
        target_value = numeric(target.get(column))
        peer_values = [value for value in peer_df[column].map(numeric).dropna().tolist()]
        if target_value is None:
            if peer_values:
                adoption = 100.0 * len(peer_values) / max(len(peer_df), 1)
                missing_kpis.append(
                    {
                        "kpi": meta["label"],
                        "peer_disclosure_rate_percent": round(adoption, 2),
                        "peer_count_with_value": len(peer_values),
                        "suggestion": f"Track and disclose {meta['label'].lower()} to match peer reporting practice.",
                    }
                )
            continue
        if not peer_values:
            comparisons.append(
                {
                    "kpi": meta["label"],
                    "company_value": target_value,
                    "unit": meta["unit"],
                    "status": "No peer values available for comparison",
                }
            )
            continue

        stats = compare_kpi(target_value, peer_values, meta["lower_is_better"])
        comparisons.append(
            {
                "kpi": meta["label"],
                "unit": meta["unit"],
                "company_value": target_value,
                "lower_is_better": meta["lower_is_better"],
                **stats,
            }
        )

    disclosure_gaps: list[dict[str, Any]] = []
    for column, label in DISCLOSURE_LABELS.items():
        if column not in df.columns:
            continue
        target_has = truthy(target.get(column))
        if target_has or peer_df.empty:
            continue
        adoption = 100.0 * peer_df[column].map(truthy).sum() / len(peer_df)
        if adoption >= adoption_threshold:
            examples = (
                peer_df[peer_df[column].map(truthy)]["company"].dropna().astype(str).drop_duplicates().head(5).tolist()
            )
            disclosure_gaps.append(
                {
                    "disclosure": label,
                    "peer_adoption_percent": round(adoption, 2),
                    "peer_examples": examples,
                    "suggestion": f"Add a clearer {label.lower()} disclosure because it is common in the selected peer group.",
                }
            )

    target_meta = {
        "company": clean_text(target["company"]),
        "reporting_year": clean_text(target["reporting_year"]),
        "sector": clean_text(target["sector"]),
        "benchmark_quality_score": numeric(target.get("benchmark_quality_score")),
    }
    return {
        "target": target_meta,
        "peer_group": {
            "type": peer_group_label,
            "companies": int(len(peer_df)),
            "sample_companies": peer_df["company"].dropna().astype(str).drop_duplicates().head(10).tolist(),
        },
        "kpi_comparisons": comparisons,
        "missing_kpis": missing_kpis,
        "disclosure_gaps": disclosure_gaps,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    target = report["target"]
    lines = [
        f"# Comparative ESG Benchmark: {target['company']}",
        "",
        f"- Year: {target['reporting_year']}",
        f"- Sector: {target['sector']}",
        f"- Peer group: {report['peer_group']['type']}",
        f"- Peer companies used: {report['peer_group']['companies']}",
        "",
        "## KPI Comparisons",
        "",
    ]
    if report["kpi_comparisons"]:
        lines.append("| KPI | Company value | Peer median | Peer average | Peer range | Rank | Interpretation |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for item in report["kpi_comparisons"]:
            unit = item.get("unit", "")
            rank = (
                f"{item.get('rank_among_group_including_company')}/"
                f"{item.get('group_size_including_company')}"
                if item.get("rank_among_group_including_company")
                else "-"
            )
            lines.append(
                "| {kpi} | {company} | {median} | {average} | {minv} to {maxv} | {rank} | {interp} |".format(
                    kpi=item["kpi"],
                    company=format_value(item.get("company_value"), unit),
                    median=format_value(item.get("peer_median"), unit),
                    average=format_value(item.get("peer_average"), unit),
                    minv=format_value(item.get("peer_min"), unit),
                    maxv=format_value(item.get("peer_max"), unit),
                    rank=rank,
                    interp=item.get("interpretation", item.get("status", "-")),
                )
            )
    else:
        lines.append("No KPI comparisons were available.")

    lines.extend(["", "## Missing KPI Opportunities", ""])
    if report["missing_kpis"]:
        for item in report["missing_kpis"]:
            lines.append(
                f"- {item['kpi']}: disclosed by {item['peer_disclosure_rate_percent']}% of selected peers. "
                f"{item['suggestion']}"
            )
    else:
        lines.append("No missing KPI opportunities were detected from available peer data.")

    lines.extend(["", "## Disclosure Gaps", ""])
    if report["disclosure_gaps"]:
        for item in report["disclosure_gaps"]:
            examples = ", ".join(item["peer_examples"]) if item["peer_examples"] else "No examples available"
            lines.append(
                f"- {item['disclosure']}: {item['peer_adoption_percent']}% peer adoption. "
                f"Examples: {examples}. {item['suggestion']}"
            )
    else:
        lines.append("No high-adoption disclosure gaps were detected.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", required=True, help="Target company name/symbol")
    parser.add_argument("--year", help="Reporting year, e.g. FY 2024-25")
    parser.add_argument("--sector", help="Override sector peer group")
    parser.add_argument("--peers", help="Comma-separated custom peer group")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--adoption-threshold", type=float, default=60.0)
    args = parser.parse_args()

    df = load_company_year(args.data_dir)
    target = pick_target(df, args.company, args.year)
    peer_df, peer_group_label = select_peers(df, target, args.sector, args.peers, args.year)
    report = build_report(df, target, peer_df, peer_group_label, args.adoption_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_company = re.sub(r"[^A-Za-z0-9]+", "_", clean_text(report["target"]["company"])).strip("_") or "company"
    json_path = args.output_dir / f"comparative_benchmark_{safe_company}.json"
    md_path = args.output_dir / f"comparative_benchmark_{safe_company}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print(json.dumps(report["target"], indent=2))
    print(f"Peer group: {report['peer_group']['type']} ({report['peer_group']['companies']} companies)")
    print(f"KPI comparisons: {len(report['kpi_comparisons'])}")
    print(f"Missing KPI opportunities: {len(report['missing_kpis'])}")
    print(f"Disclosure gaps: {len(report['disclosure_gaps'])}")
    print("Wrote:", json_path)
    print("Wrote:", md_path)


if __name__ == "__main__":
    main()
