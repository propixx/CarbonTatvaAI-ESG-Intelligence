#!/usr/bin/env python3
"""Convert the uploaded ESG/BRSR master CSV into instruction-tuning JSON."""

import argparse
import csv
import json
import re
from pathlib import Path


BOOL_FIELDS = [
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

SCORE_FIELDS = [
    "score_environmental",
    "score_social",
    "score_governance",
    "score_climate_risk",
    "score_net_zero",
    "score_energy",
    "score_water",
    "score_waste",
    "score_scope_1",
    "score_scope_2",
    "score_scope_3",
    "score_diversity",
    "score_human_rights",
    "score_csr",
    "score_supply_chain",
    "score_board_governance",
    "score_tcfd",
    "score_ifrs_s1_s2",
    "score_cdp",
]

KPI_FIELDS = [
    "kpi_scope1_emissions_tco2e_current",
    "kpi_scope1_emissions_tco2e_previous",
    "kpi_scope1_emissions_yoy_reduction_percent",
    "kpi_scope2_emissions_tco2e_current",
    "kpi_scope2_emissions_tco2e_previous",
    "kpi_scope2_emissions_yoy_reduction_percent",
    "kpi_scope3_emissions_tco2e_current",
    "kpi_scope3_emissions_tco2e_previous",
    "kpi_scope3_emissions_yoy_reduction_percent",
    "kpi_scope1_scope2_total_tco2e_current",
    "kpi_scope1_scope2_total_tco2e_previous",
    "kpi_scope1_scope2_yoy_reduction_percent",
    "kpi_renewable_energy_percent",
    "kpi_total_energy_consumption_gj",
    "kpi_water_consumption_kl_current",
    "kpi_total_waste_generated_current",
    "kpi_waste_recycled_percent",
    "kpi_female_employee_percent",
    "kpi_women_on_board_percent",
    "kpi_energy_intensity_current",
    "kpi_net_zero_target_year",
]

EVIDENCE_FIELDS = [
    "kpi_scope1_evidence",
    "kpi_scope2_evidence",
    "kpi_scope3_evidence",
    "kpi_renewable_energy_evidence",
    "kpi_water_consumption_evidence",
    "kpi_waste_recycled_evidence",
    "kpi_total_waste_generated_evidence",
    "kpi_female_employee_evidence",
    "kpi_energy_intensity_evidence",
]


def clean(value: str | None) -> str:
    value = "" if value is None else str(value)
    value = value.replace("\ufeff", "").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def has_value(value: str | None) -> bool:
    return clean(value) not in {"", "nan", "NaN", "None", "null"}


def fmt_num(value: str | None, suffix: str = "") -> str:
    value = clean(value)
    if not value:
        return "not disclosed"
    try:
        number = float(value.replace(",", ""))
    except ValueError:
        return value
    rendered = f"{number:,.2f}".rstrip("0").rstrip(".")
    return f"{rendered}{suffix}"


def label(field: str) -> str:
    return field.replace("kpi_", "").replace("score_", "").replace("has_", "").replace("_", " ")


def truthy(value: str | None) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def kv_block(row: dict[str, str], fields: list[str]) -> str:
    lines = []
    for field in fields:
        value = clean(row.get(field))
        if value:
            lines.append(f"- {label(field)}: {value}")
    return "\n".join(lines)


def truncate(value: str, limit: int = 900) -> str:
    value = clean(value)
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def base_context(row: dict[str, str]) -> str:
    present = [label(field) for field in BOOL_FIELDS if truthy(row.get(field))]
    missing = [label(field) for field in BOOL_FIELDS if clean(row.get(field)) and not truthy(row.get(field))]
    pieces = [
        f"Company: {clean(row.get('company'))}",
        f"Reporting year: {clean(row.get('reporting_year'))}",
        f"Sector: {clean(row.get('meta_sector'))}",
        f"Market cap: {clean(row.get('meta_market_cap'))}",
        f"Framework: {clean(row.get('meta_framework_used'))}",
        f"Assurance: {clean(row.get('meta_assurance_type'))}",
        f"Geography: {clean(row.get('meta_geography'))}",
        f"Top sections: {clean(row.get('top_sections'))}",
        f"Present disclosure areas: {', '.join(present) if present else 'not identified'}",
        f"Missing disclosure areas: {', '.join(missing) if missing else 'none identified'}",
    ]
    kpis = kv_block(row, KPI_FIELDS)
    if kpis:
        pieces.append("KPIs:\n" + kpis)
    scores = kv_block(row, SCORE_FIELDS)
    if scores:
        pieces.append("Disclosure scores:\n" + scores)
    return "\n".join(piece for piece in pieces if piece and not piece.endswith(": "))


def summary_output(row: dict[str, str]) -> str:
    summary = clean(row.get("llm_training_summary"))
    if summary:
        return summary
    return (
        f"{clean(row.get('company'))} reported for {clean(row.get('reporting_year'))} in the "
        f"{clean(row.get('meta_sector'))} sector. The disclosure uses "
        f"{clean(row.get('meta_framework_used')) or 'an ESG reporting framework'} and covers "
        f"{clean(row.get('top_sections')) or 'the identified ESG sections'}."
    )


def carbon_output(row: dict[str, str]) -> str:
    company = clean(row.get("company"))
    year = clean(row.get("reporting_year"))
    lines = [
        f"For {company} in {year}, the available GHG emissions data indicates:",
        f"- Scope 1 emissions: {fmt_num(row.get('kpi_scope1_emissions_tco2e_current'), ' tCO2e')} current versus {fmt_num(row.get('kpi_scope1_emissions_tco2e_previous'), ' tCO2e')} previous, with YoY reduction of {fmt_num(row.get('kpi_scope1_emissions_yoy_reduction_percent'), '%')}.",
        f"- Scope 2 emissions: {fmt_num(row.get('kpi_scope2_emissions_tco2e_current'), ' tCO2e')} current versus {fmt_num(row.get('kpi_scope2_emissions_tco2e_previous'), ' tCO2e')} previous, with YoY reduction of {fmt_num(row.get('kpi_scope2_emissions_yoy_reduction_percent'), '%')}.",
    ]
    if has_value(row.get("kpi_scope3_emissions_tco2e_current")):
        lines.append(
            f"- Scope 3 emissions: {fmt_num(row.get('kpi_scope3_emissions_tco2e_current'), ' tCO2e')} current versus {fmt_num(row.get('kpi_scope3_emissions_tco2e_previous'), ' tCO2e')} previous, with YoY reduction of {fmt_num(row.get('kpi_scope3_emissions_yoy_reduction_percent'), '%')}."
        )
    if has_value(row.get("kpi_scope1_scope2_total_tco2e_current")):
        lines.append(
            f"- Combined Scope 1 and Scope 2 emissions: {fmt_num(row.get('kpi_scope1_scope2_total_tco2e_current'), ' tCO2e')} current versus {fmt_num(row.get('kpi_scope1_scope2_total_tco2e_previous'), ' tCO2e')} previous, with YoY reduction of {fmt_num(row.get('kpi_scope1_scope2_yoy_reduction_percent'), '%')}."
        )
    if has_value(row.get("kpi_scope1_evidence")) or has_value(row.get("kpi_scope2_evidence")):
        evidence = " ".join(
            truncate(row.get(field, ""), 420)
            for field in ("kpi_scope1_evidence", "kpi_scope2_evidence", "kpi_scope3_evidence")
            if has_value(row.get(field))
        )
        lines.append(f"Evidence note: {truncate(evidence, 900)}")
    return "\n".join(lines)


def coverage_output(row: dict[str, str]) -> str:
    present = [label(field) for field in BOOL_FIELDS if truthy(row.get(field))]
    absent = [label(field) for field in BOOL_FIELDS if clean(row.get(field)) and not truthy(row.get(field))]
    strongest = sorted(
        ((int(float(clean(row.get(field)) or 0)), label(field)) for field in SCORE_FIELDS),
        reverse=True,
    )[:5]
    lines = [
        f"The disclosure for {clean(row.get('company'))} covers {len(present)} ESG topic areas.",
        f"Covered areas include: {', '.join(present) if present else 'none identified'}.",
        f"Gaps or absent areas include: {', '.join(absent) if absent else 'none identified'}.",
    ]
    if strongest:
        lines.append(
            "Highest-scoring disclosure areas are: "
            + ", ".join(f"{name} ({score})" for score, name in strongest if score > 0)
            + "."
        )
    return "\n".join(lines)


def make_examples(row: dict[str, str], mode: str) -> list[dict[str, str]]:
    context = base_context(row)
    examples = []
    if mode in {"all", "summary"}:
        examples.append(
            {
                "instruction": "Prepare a structured ESG disclosure summary for this company.",
                "input": context,
                "output": summary_output(row),
            }
        )
    if mode in {"all", "carbon"} and (
        has_value(row.get("kpi_scope1_emissions_tco2e_current"))
        or has_value(row.get("kpi_scope2_emissions_tco2e_current"))
        or has_value(row.get("kpi_scope3_emissions_tco2e_current"))
    ):
        carbon_context = context + "\n\nEmission evidence:\n" + "\n".join(
            f"- {label(field)}: {truncate(row.get(field, ''), 700)}"
            for field in EVIDENCE_FIELDS
            if has_value(row.get(field))
        )
        examples.append(
            {
                "instruction": "Summarize the company's greenhouse gas emissions and year-on-year changes.",
                "input": carbon_context,
                "output": carbon_output(row),
            }
        )
    if mode in {"all", "coverage"}:
        examples.append(
            {
                "instruction": "Identify the ESG disclosure coverage and important reporting gaps.",
                "input": context,
                "output": coverage_output(row),
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/esg_prd_master_dataset_25-26.csv")
    parser.add_argument("--output", default="data/esg_prd_instruction.json")
    parser.add_argument("--mode", choices=["all", "summary", "carbon", "coverage"], default="all")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    examples: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            examples.extend(make_examples(row, args.mode))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(examples, handle, indent=2, ensure_ascii=False)

    print(f"Converted {input_path} into {len(examples)} instruction examples.")
    print(f"Saved: {output_path}")
    if examples:
        print("\nFirst example preview:")
        print(json.dumps(examples[0], indent=2, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
