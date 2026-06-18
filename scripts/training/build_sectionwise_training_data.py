#!/usr/bin/env python3
"""Build section-wise ESG/BRSR fine-tuning data from PRD master CSVs.

This is CPU-only data preparation. It does not need a GPU.

Input rows: PRD master CSVs with metadata, KPIs, and evidence columns.
Output rows: one training sample per ESG section.

Example sample:
    input  = company metadata + GHG KPIs + GHG evidence snippets
    output = clean GHG Emissions section text
"""

import argparse
import csv
import json
import math
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


def clean(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    value = str(value).replace("\ufeff", "").strip()
    return re.sub(r"\s+", " ", value)


def has_value(value):
    return clean(value) not in {"", "nan", "NaN", "None", "null", "[]", "{}"}


def as_float(value):
    value = clean(value)
    if not value:
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def fmt_num(value, decimals=2):
    number = as_float(value)
    if number is None:
        return None
    text = f"{number:,.{decimals}f}".rstrip("0").rstrip(".")
    return text


def fmt_pct(value):
    number = as_float(value)
    if number is None:
        return None
    return f"{abs(number):.2f}".rstrip("0").rstrip(".") + "%"


def yoy_sentence(metric_name, current, previous, yoy_reduction, unit):
    cur = fmt_num(current)
    prev = fmt_num(previous)
    if cur is None:
        return None
    if prev is None:
        return f"{metric_name} stood at {cur} {unit} for the reporting year."
    yoy = as_float(yoy_reduction)
    if yoy is None:
        return f"{metric_name} stood at {cur} {unit}, compared with {prev} {unit} in the previous year."
    pct = fmt_pct(yoy)
    direction = "reduced" if yoy >= 0 else "increased"
    return (
        f"{metric_name} stood at {cur} {unit}, compared with {prev} {unit} in the previous year. "
        f"{metric_name} {direction} by {pct} year-on-year."
    )


def company_name(row):
    return clean(row.get("company")) or "The company"


def reporting_year(row):
    return clean(row.get("reporting_year")) or "the reporting year"


def metadata_block(row):
    lines = []
    for field in META_FIELDS:
        value = clean(row.get(field))
        if value:
            label = field.replace("meta_", "").replace("_", " ")
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def evidence_block(row, fields):
    lines = []
    idx = 1
    for field in fields:
        value = clean(row.get(field))
        if value:
            label = field.replace("kpi_", "").replace("_", " ")
            lines.append(f"[Evidence {idx} | {label}]\n{value[:1200]}")
            idx += 1
    return "\n\n".join(lines)


def kpi_block(row, fields):
    lines = []
    for field in fields:
        value = clean(row.get(field))
        if value:
            label = field.replace("kpi_", "").replace("_", " ")
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def make_input(section_title, section_instruction, row, kpi_fields, evidence_fields):
    parts = [
        "Task:",
        f"Generate the \"{section_title}\" section of a KPI- and evidence-supported ESG/BRSR report.",
        "",
        "Section instruction:",
        section_instruction,
        "",
        "Company metadata:",
        metadata_block(row),
        "",
        "KPI data:",
        kpi_block(row, kpi_fields) or "No structured KPI values available.",
    ]
    evidence = evidence_block(row, evidence_fields)
    if evidence:
        parts.extend(["", "KPI evidence snippets:", evidence])
    return "\n".join(parts).strip()


def make_example(section, instruction, row, kpi_fields, evidence_fields, output):
    output = clean(output)
    if len(output) < 80:
        return None
    return {
        "section": section,
        "instruction": f"Generate the {section} section from the provided KPI data and evidence.",
        "input": make_input(section, instruction, row, kpi_fields, evidence_fields),
        "output": output,
        "company": company_name(row),
        "reporting_year": reporting_year(row),
    }


def ghg_section(row):
    fields = [
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
    ]
    if not any(has_value(row.get(field)) for field in fields):
        return None
    sentences = [f"{company_name(row)} disclosed its greenhouse gas emissions for {reporting_year(row)} based on the available Scope-wise emissions data."]
    for metric, cur, prev, yoy in [
        ("Scope 1 emissions", "kpi_scope1_emissions_tco2e_current", "kpi_scope1_emissions_tco2e_previous", "kpi_scope1_emissions_yoy_reduction_percent"),
        ("Scope 2 emissions", "kpi_scope2_emissions_tco2e_current", "kpi_scope2_emissions_tco2e_previous", "kpi_scope2_emissions_yoy_reduction_percent"),
        ("Scope 3 emissions", "kpi_scope3_emissions_tco2e_current", "kpi_scope3_emissions_tco2e_previous", "kpi_scope3_emissions_yoy_reduction_percent"),
        ("Combined Scope 1 and Scope 2 emissions", "kpi_scope1_scope2_total_tco2e_current", "kpi_scope1_scope2_total_tco2e_previous", "kpi_scope1_scope2_yoy_reduction_percent"),
    ]:
        sentence = yoy_sentence(metric, row.get(cur), row.get(prev), row.get(yoy), "tCO2e")
        if sentence:
            sentences.append(sentence)
    sentences.append("The disclosure is limited to the emissions KPIs and evidence available in the input data, and no unsupported claims have been made on climate targets, offsets, or external frameworks.")
    return make_example(
        "GHG Emissions",
        "Generate the GHG Emissions section covering Scope 1, Scope 2, Scope 3 and year-on-year movement where available.",
        row,
        fields,
        ["kpi_scope1_evidence", "kpi_scope2_evidence", "kpi_scope3_evidence"],
        " ".join(sentences),
    )


def water_section(row):
    fields = [
        "kpi_water_consumption_kl_current",
        "kpi_water_consumption_kl_previous",
        "kpi_water_consumption_yoy_reduction_percent",
        "kpi_water_withdrawal_kl_current",
        "kpi_water_withdrawal_kl_previous",
        "kpi_water_withdrawal_yoy_reduction_percent",
    ]
    if not any(has_value(row.get(field)) for field in fields):
        return None
    sentences = [f"{company_name(row)} reported water-related performance indicators for {reporting_year(row)} based on the available BRSR disclosures."]
    for metric, cur, prev, yoy in [
        ("Water consumption", "kpi_water_consumption_kl_current", "kpi_water_consumption_kl_previous", "kpi_water_consumption_yoy_reduction_percent"),
        ("Water withdrawal", "kpi_water_withdrawal_kl_current", "kpi_water_withdrawal_kl_previous", "kpi_water_withdrawal_yoy_reduction_percent"),
    ]:
        sentence = yoy_sentence(metric, row.get(cur), row.get(prev), row.get(yoy), "KL")
        if sentence:
            sentences.append(sentence)
    sentences.append("The section is based only on the water KPIs and evidence available in the input data.")
    return make_example(
        "Water Management",
        "Generate the Water Management section covering water consumption, water withdrawal, and year-on-year movement where available.",
        row,
        fields,
        ["kpi_water_consumption_evidence"],
        " ".join(sentences),
    )


def waste_section(row):
    fields = [
        "kpi_total_waste_generated_current",
        "kpi_total_waste_generated_previous",
        "kpi_waste_recycled_current",
        "kpi_waste_recycled_previous",
        "kpi_waste_recycled_unit",
        "kpi_waste_recycled_percent",
    ]
    if not any(has_value(row.get(field)) for field in fields):
        return None
    company = company_name(row)
    year = reporting_year(row)
    sentences = [f"{company} disclosed waste-related performance for {year} using the available waste generation and recycling indicators."]
    total_current = fmt_num(row.get("kpi_total_waste_generated_current"))
    total_previous = fmt_num(row.get("kpi_total_waste_generated_previous"))
    if total_current:
        if total_previous:
            sentences.append(f"Total waste generated stood at {total_current}, compared with {total_previous} in the previous year.")
        else:
            sentences.append(f"Total waste generated stood at {total_current} for the reporting year.")
    recycled_current = fmt_num(row.get("kpi_waste_recycled_current"))
    recycled_previous = fmt_num(row.get("kpi_waste_recycled_previous"))
    unit = clean(row.get("kpi_waste_recycled_unit")) or "units"
    recycled_pct = fmt_num(row.get("kpi_waste_recycled_percent"))
    if recycled_current:
        sentence = f"Waste recycled was {recycled_current} {unit}"
        if recycled_previous:
            sentence += f", compared with {recycled_previous} {unit} in the previous year"
        if recycled_pct:
            sentence += f", representing {recycled_pct}% of the relevant waste indicator"
        sentences.append(sentence + ".")
    sentences.append("The section avoids unsupported claims beyond the waste KPIs and evidence provided.")
    return make_example(
        "Waste Management",
        "Generate the Waste Management section covering waste generated, waste recycled, and available recycling indicators.",
        row,
        fields,
        ["kpi_total_waste_generated_evidence", "kpi_waste_recycled_evidence"],
        " ".join(sentences),
    )


def energy_section(row):
    fields = [
        "kpi_renewable_energy_percent",
        "kpi_renewable_energy_consumption_gj",
        "kpi_total_energy_consumption_gj",
        "kpi_energy_intensity_current",
        "kpi_energy_intensity_previous",
        "kpi_energy_intensity_unit",
        "kpi_energy_intensity_yoy_reduction_percent",
    ]
    if not any(has_value(row.get(field)) for field in fields):
        return None
    company = company_name(row)
    year = reporting_year(row)
    sentences = [f"{company} reported energy-related indicators for {year} based on the available energy consumption and intensity data."]
    total = fmt_num(row.get("kpi_total_energy_consumption_gj"))
    if total:
        sentences.append(f"Total energy consumption was {total} GJ for the reporting year.")
    renewable_pct = fmt_num(row.get("kpi_renewable_energy_percent"))
    renewable_gj = fmt_num(row.get("kpi_renewable_energy_consumption_gj"))
    if renewable_pct:
        sentences.append(f"Renewable energy represented {renewable_pct}% of the relevant energy indicator.")
    if renewable_gj:
        sentences.append(f"Renewable energy consumption was {renewable_gj} GJ.")
    intensity_unit = clean(row.get("kpi_energy_intensity_unit")) or "reported unit"
    sentence = yoy_sentence(
        "Energy intensity",
        row.get("kpi_energy_intensity_current"),
        row.get("kpi_energy_intensity_previous"),
        row.get("kpi_energy_intensity_yoy_reduction_percent"),
        intensity_unit,
    )
    if sentence:
        sentences.append(sentence)
    sentences.append("The section is limited to the energy KPIs and evidence available in the input data.")
    return make_example(
        "Energy Management",
        "Generate the Energy Management section covering total energy, renewable energy and energy intensity where available.",
        row,
        fields,
        ["kpi_renewable_energy_evidence", "kpi_energy_intensity_evidence"],
        " ".join(sentences),
    )


def diversity_section(row):
    fields = ["kpi_female_employee_percent", "kpi_women_on_board_percent"]
    if not any(has_value(row.get(field)) for field in fields):
        return None
    company = company_name(row)
    year = reporting_year(row)
    sentences = [f"{company} disclosed diversity-related indicators for {year} based on the available workforce and board composition data."]
    female = fmt_num(row.get("kpi_female_employee_percent"))
    board = fmt_num(row.get("kpi_women_on_board_percent"))
    if female:
        sentences.append(f"Female employees represented {female}% of the workforce indicator disclosed in the dataset.")
    if board:
        sentences.append(f"Women on the Board represented {board}% of the board composition indicator disclosed in the dataset.")
    sentences.append("The disclosure is limited to the diversity KPIs and evidence available in the input data.")
    return make_example(
        "Diversity and Inclusion",
        "Generate the Diversity and Inclusion section using female employee and women-on-board indicators where available.",
        row,
        fields,
        ["kpi_female_employee_evidence"],
        " ".join(sentences),
    )


SECTION_BUILDERS = [ghg_section, water_section, waste_section, energy_section, diversity_section]


def read_rows(csv_paths):
    for path in csv_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["__source_file"] = str(path)
                yield row


def find_csvs(paths):
    csvs = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix.lower() == ".csv" and "audit" not in path.name.lower():
            csvs.append(path)
        elif path.is_dir():
            csvs.extend(
                p for p in path.rglob("esg_prd_master_dataset*.csv")
                if "audit" not in p.name.lower()
            )
    return sorted(set(csvs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="PRD master CSV files or directories.")
    parser.add_argument("--output", default="data/sectionwise_training_dataset.json")
    parser.add_argument("--jsonl", default="data/sectionwise_training_dataset.jsonl")
    parser.add_argument("--manifest", default="data/sectionwise_training_manifest.json")
    args = parser.parse_args()

    csv_paths = find_csvs(args.paths)
    if not csv_paths:
        raise FileNotFoundError("No PRD master CSV files found.")

    print("Using CSV files:")
    for path in csv_paths:
        print("-", path)

    examples = []
    section_counts = {}
    seen = set()
    row_count = 0
    for row in read_rows(csv_paths):
        row_count += 1
        for builder in SECTION_BUILDERS:
            example = builder(row)
            if not example:
                continue
            key = (example["section"], example["input"], example["output"])
            if key in seen:
                continue
            seen.add(key)
            examples.append(example)
            section_counts[example["section"]] = section_counts.get(example["section"], 0) + 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(examples, handle, indent=2, ensure_ascii=False)

    jsonl_path = Path(args.jsonl)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    manifest = {
        "source_files": [str(path) for path in csv_paths],
        "source_rows": row_count,
        "total_examples": len(examples),
        "section_counts": section_counts,
    }
    manifest_path = Path(args.manifest)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(f"\nSource rows: {row_count}")
    print(f"Generated examples: {len(examples)}")
    print("Section counts:", section_counts)
    print(f"Saved JSON: {output_path}")
    print(f"Saved JSONL: {jsonl_path}")
    print(f"Saved manifest: {manifest_path}")
    if examples:
        print("\nSample:")
        print(json.dumps(examples[0], indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
