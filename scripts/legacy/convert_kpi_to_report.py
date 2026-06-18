#!/usr/bin/env python3
"""Convert ESG KPI CSV rows into KPI-to-report generation examples.

Core task:
    input  = company name + KPI / disclosure data
    output = ESG report-style narrative text

Only company, disclosure, and KPI fields are used as input. The target text is
the ESG report narrative stored in ``llm_training_summary``.
"""

import argparse
import csv
import json
import re
from pathlib import Path


META_FIELDS = [
    "company",
    "reporting_year",
    "meta_sector",
    "meta_market_cap",
    "meta_framework_used",
    "meta_brsr_version",
    "meta_assurance_type",
    "meta_geography",
]

DISCLOSURE_FLAG_FIELDS = [
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

KPI_FIELDS = [
    "top_sections",
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
    "kpi_renewable_energy_consumption_gj",
    "kpi_total_energy_consumption_gj",
    "kpi_water_consumption_kl_current",
    "kpi_water_consumption_kl_previous",
    "kpi_water_consumption_yoy_reduction_percent",
    "kpi_water_withdrawal_kl_current",
    "kpi_water_withdrawal_kl_previous",
    "kpi_water_withdrawal_yoy_reduction_percent",
    "kpi_total_waste_generated_current",
    "kpi_total_waste_generated_previous",
    "kpi_waste_recycled_current",
    "kpi_waste_recycled_previous",
    "kpi_waste_recycled_unit",
    "kpi_waste_recycled_percent",
    "kpi_female_employee_percent",
    "kpi_women_on_board_percent",
    "kpi_energy_intensity_current",
    "kpi_energy_intensity_previous",
    "kpi_energy_intensity_unit",
    "kpi_energy_intensity_yoy_reduction_percent",
    "kpi_net_zero_target_year",
    "kpi_targets_count",
    "kpi_direct_yoy_reductions_count",
]


def clean(value: str | None) -> str:
    value = "" if value is None else str(value)
    return re.sub(r"\s+", " ", value.replace("\ufeff", "")).strip()


def has_value(value: str | None) -> bool:
    return clean(value) not in {"", "nan", "NaN", "None", "null", "[]", "{}"}


def label(field: str) -> str:
    return field.replace("meta_", "").replace("kpi_", "").replace("has_", "").replace("_", " ")


def build_input(row: dict[str, str]) -> str:
    lines = []
    for field in META_FIELDS + DISCLOSURE_FLAG_FIELDS + KPI_FIELDS:
        value = clean(row.get(field))
        if has_value(value):
            lines.append(f"{label(field)}: {value}")
    return "\n".join(lines)


def make_example(row: dict[str, str]) -> dict[str, str] | None:
    output = clean(row.get("llm_training_summary"))
    if not output:
        return None
    company = clean(row.get("company")) or "the company"
    year = clean(row.get("reporting_year"))
    instruction = (
        "Generate a professional ESG report narrative from the provided company name "
        "and KPI data. Write report-style disclosure text, not a bullet summary."
    )
    if year:
        instruction += f" Use the reporting period {year} where relevant."
    return {
        "instruction": instruction,
        "input": build_input(row),
        "output": output,
        "company": company,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/esg_prd_master_dataset_25-26.csv")
    parser.add_argument("--output", default="data/kpi_to_esg_report.json")
    parser.add_argument("--keep-company-field", action="store_true")
    args = parser.parse_args()

    examples = []
    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            example = make_example(row)
            if example:
                if not args.keep_company_field:
                    example.pop("company", None)
                examples.append(example)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(examples, handle, indent=2, ensure_ascii=False)

    print(f"Saved {len(examples)} KPI-to-report examples to {output_path}")
    if examples:
        print(json.dumps(examples[0], indent=2, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
