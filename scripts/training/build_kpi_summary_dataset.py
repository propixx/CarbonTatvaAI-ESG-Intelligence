#!/usr/bin/env python3
"""Build company-level KPI-to-ESG-summary fine-tuning data.

The source PRD tables contain extracted metadata, KPIs, evidence, intent labels,
and an ``llm_training_summary`` audit field. This builder intentionally uses
only company metadata and scalar KPI columns. It creates a fresh, factual
narrative target for each company and never trains on the audit summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are CarbonTatvaAI, an ESG reporting analyst. Write a concise, factual "
    "ESG narrative summary using only the supplied company metadata and KPI "
    "data. Mention all material KPI values that are provided, preserve units, "
    "and describe year-on-year movement accurately. Do not invent policies, "
    "initiatives, awards, targets, committees, certifications, framework "
    "alignment, or explanations that are absent from the input. The summary is "
    "a drafting aid, not a complete statutory BRSR report."
)

META_FIELDS = {
    "company": "Company",
    "reporting_year": "Reporting year",
    "meta_sector": "Sector",
    "meta_market_cap": "Market capitalisation",
    "meta_framework_used": "Reporting framework",
    "meta_brsr_version": "BRSR version",
    "meta_assurance_type": "Assurance",
    "meta_geography": "Geography",
}

EXCLUDED_KPI_SUFFIXES = (
    "_evidence",
    "_source",
    "_json",
)

NUMERIC_KPI_FIELDS = [
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
    "kpi_waste_recycled_percent",
    "kpi_female_employee_percent",
    "kpi_women_on_board_percent",
    "kpi_energy_intensity_current",
    "kpi_energy_intensity_previous",
    "kpi_energy_intensity_yoy_reduction_percent",
]

TEXT_KPI_FIELDS = [
    "kpi_waste_recycled_unit",
    "kpi_energy_intensity_unit",
    "kpi_net_zero_target_year",
]

PERCENTAGE_LEVEL_FIELDS = {
    "kpi_renewable_energy_percent",
    "kpi_waste_recycled_percent",
    "kpi_female_employee_percent",
    "kpi_women_on_board_percent",
}

YOY_SPECS = {
    "scope1": (
        "kpi_scope1_emissions_tco2e_current",
        "kpi_scope1_emissions_tco2e_previous",
        "kpi_scope1_emissions_yoy_reduction_percent",
    ),
    "scope2": (
        "kpi_scope2_emissions_tco2e_current",
        "kpi_scope2_emissions_tco2e_previous",
        "kpi_scope2_emissions_yoy_reduction_percent",
    ),
    "scope3": (
        "kpi_scope3_emissions_tco2e_current",
        "kpi_scope3_emissions_tco2e_previous",
        "kpi_scope3_emissions_yoy_reduction_percent",
    ),
    "scope1_scope2_total": (
        "kpi_scope1_scope2_total_tco2e_current",
        "kpi_scope1_scope2_total_tco2e_previous",
        "kpi_scope1_scope2_yoy_reduction_percent",
    ),
    "water_consumption": (
        "kpi_water_consumption_kl_current",
        "kpi_water_consumption_kl_previous",
        "kpi_water_consumption_yoy_reduction_percent",
    ),
    "water_withdrawal": (
        "kpi_water_withdrawal_kl_current",
        "kpi_water_withdrawal_kl_previous",
        "kpi_water_withdrawal_yoy_reduction_percent",
    ),
    "energy_intensity": (
        "kpi_energy_intensity_current",
        "kpi_energy_intensity_previous",
        "kpi_energy_intensity_yoy_reduction_percent",
    ),
}

FIELD_LABELS = {
    "kpi_scope1_emissions_tco2e_current": "Scope 1 emissions, current year (tCO2e)",
    "kpi_scope1_emissions_tco2e_previous": "Scope 1 emissions, previous year (tCO2e)",
    "kpi_scope1_emissions_yoy_reduction_percent": "Scope 1 year-on-year reduction convention (%)",
    "kpi_scope2_emissions_tco2e_current": "Scope 2 emissions, current year (tCO2e)",
    "kpi_scope2_emissions_tco2e_previous": "Scope 2 emissions, previous year (tCO2e)",
    "kpi_scope2_emissions_yoy_reduction_percent": "Scope 2 year-on-year reduction convention (%)",
    "kpi_scope3_emissions_tco2e_current": "Scope 3 emissions, current year (tCO2e)",
    "kpi_scope3_emissions_tco2e_previous": "Scope 3 emissions, previous year (tCO2e)",
    "kpi_scope3_emissions_yoy_reduction_percent": "Scope 3 year-on-year reduction convention (%)",
    "kpi_scope1_scope2_total_tco2e_current": "Combined Scope 1 and Scope 2 emissions, current year (tCO2e)",
    "kpi_scope1_scope2_total_tco2e_previous": "Combined Scope 1 and Scope 2 emissions, previous year (tCO2e)",
    "kpi_scope1_scope2_yoy_reduction_percent": "Combined Scope 1 and Scope 2 year-on-year reduction convention (%)",
    "kpi_renewable_energy_percent": "Renewable energy share (%)",
    "kpi_renewable_energy_consumption_gj": "Renewable energy consumption (GJ)",
    "kpi_total_energy_consumption_gj": "Total energy consumption (GJ)",
    "kpi_water_consumption_kl_current": "Water consumption, current year (KL)",
    "kpi_water_consumption_kl_previous": "Water consumption, previous year (KL)",
    "kpi_water_consumption_yoy_reduction_percent": "Water consumption year-on-year reduction convention (%)",
    "kpi_water_withdrawal_kl_current": "Water withdrawal, current year (KL)",
    "kpi_water_withdrawal_kl_previous": "Water withdrawal, previous year (KL)",
    "kpi_water_withdrawal_yoy_reduction_percent": "Water withdrawal year-on-year reduction convention (%)",
    "kpi_total_waste_generated_current": "Total waste generated, current year",
    "kpi_total_waste_generated_previous": "Total waste generated, previous year",
    "kpi_waste_recycled_current": "Waste recycled or recovered, current year",
    "kpi_waste_recycled_previous": "Waste recycled or recovered, previous year",
    "kpi_waste_recycled_unit": "Waste recycled or recovered unit",
    "kpi_waste_recycled_percent": "Waste recycled or recovered share (%)",
    "kpi_female_employee_percent": "Female employees (%)",
    "kpi_women_on_board_percent": "Women on the Board (%)",
    "kpi_energy_intensity_current": "Energy intensity, current year",
    "kpi_energy_intensity_previous": "Energy intensity, previous year",
    "kpi_energy_intensity_unit": "Energy intensity unit",
    "kpi_energy_intensity_yoy_reduction_percent": "Energy intensity year-on-year reduction convention (%)",
    "kpi_net_zero_target_year": "Disclosed net-zero target year",
}


@dataclass
class SourceRow:
    row: dict[str, Any]
    source_file: str
    source_dataset: str
    company: str
    company_key: str
    nse_symbol: str


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).replace("\ufeff", "").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def is_missing(value: Any) -> bool:
    return clean_text(value).lower() in {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "unknown",
        "not mentioned",
        "not available",
        "not disclosed",
        "[]",
        "{}",
    }


def as_number(value: Any) -> float | None:
    if is_missing(value):
        return None
    text = clean_text(value).replace(",", "").replace("%", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def format_number(value: Any) -> str | None:
    number = as_number(value)
    if number is None:
        return None
    if abs(number) != 0 and abs(number) < 0.01:
        return f"{number:.8f}".rstrip("0").rstrip(".")
    if abs(number - round(number)) < 1e-9:
        return f"{int(round(number)):,}"
    return f"{number:,.2f}".rstrip("0").rstrip(".")


def format_percent(value: Any) -> str | None:
    number = as_number(value)
    if number is None:
        return None
    return f"{abs(number):.2f}".rstrip("0").rstrip(".") + "%"


def normalize_source_key(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_company_key(value: Any) -> str:
    text = clean_text(value).upper()
    text = re.sub(
        r"\b(THE|LIMITED|LTD|PRIVATE|PVT|INDIA|COMPANY|CO|CORPORATION|CORP)\b",
        " ",
        text,
    )
    return re.sub(r"[^A-Z0-9]+", "", text)


def title_company_name(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if "_" in text or re.search(r"\d{6,}", text):
        first = re.split(r"[_\s-]+", text)[0]
        return first.upper() if first else text
    if text.isupper():
        small_words = {"and", "of", "the", "for", "in"}
        words = []
        for index, word in enumerate(text.lower().split()):
            words.append(word if index and word in small_words else word.capitalize())
        return " ".join(words)
    return text


def normalize_reporting_year(value: Any, default: str = "FY 2024-25") -> str:
    text = clean_text(value).replace("\u2013", "-").replace("\u2014", "-")
    match = re.search(r"(20\d{2})\s*[-/]\s*(\d{2})", text)
    if match:
        first = int(match.group(1))
        second = int("20" + match.group(2))
        if second == first + 1 and 2020 <= first <= 2030:
            return f"FY {first}-{str(second)[-2:]}"
    return default


def source_label(path: Path) -> str:
    name = path.name.lower()
    if "24_25" in name or "24-25" in name:
        return "BRSR 2024-25 source collection"
    if "25_26" in name or "25-26" in name:
        return "BRSR 2025-26 source collection"
    return path.stem


def load_company_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    mapping: dict[str, dict[str, str]] = {}
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        for source_key, item in raw.items():
            company = clean_text(item.get("company"))
            if source_key and company:
                mapping[normalize_source_key(source_key)] = {
                    "company": company,
                    "nse_symbol": clean_text(item.get("nse_symbol")),
                }
        return mapping
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            source_key = normalize_source_key(item.get("source_key"))
            company = clean_text(item.get("company"))
            if source_key and company:
                mapping[source_key] = {
                    "company": company,
                    "nse_symbol": clean_text(item.get("nse_symbol")),
                }
    return mapping


def find_csvs(paths: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() == ".csv":
            found.append(path)
        elif path.is_dir():
            found.extend(
                item
                for item in path.rglob("*.csv")
                if "audit" not in item.name.lower()
                and ("24_25" in item.name or "24-25" in item.name or "25-26" in item.name or "25_26" in item.name)
            )
    return sorted(set(found))


def valid_kpi_value(field: str, value: Any) -> float | str | None:
    if field in TEXT_KPI_FIELDS:
        text = clean_text(value)
        if is_missing(text):
            return None
        if field == "kpi_net_zero_target_year":
            match = re.search(r"\b(20\d{2})\b", text)
            if not match or not 2025 <= int(match.group(1)) <= 2100:
                return None
            return match.group(1)
        return text[:160]

    number = as_number(value)
    if number is None:
        return None
    if field in PERCENTAGE_LEVEL_FIELDS and not 0 <= number <= 100:
        return None
    if "_yoy_reduction_percent" in field:
        return number if abs(number) <= 10000 else None
    return number if number >= 0 else None


def compute_yoy(current: Any, previous: Any) -> float | None:
    current_number = as_number(current)
    previous_number = as_number(previous)
    if current_number is None or previous_number in {None, 0}:
        return None
    return (previous_number - current_number) / abs(previous_number) * 100


def clean_kpis(row: dict[str, Any]) -> dict[str, Any]:
    kpis: dict[str, Any] = {}
    for field in NUMERIC_KPI_FIELDS + TEXT_KPI_FIELDS:
        value = valid_kpi_value(field, row.get(field))
        if value is not None:
            kpis[field] = value

    scope1_current = kpis.get("kpi_scope1_emissions_tco2e_current")
    scope2_current = kpis.get("kpi_scope2_emissions_tco2e_current")
    scope1_previous = kpis.get("kpi_scope1_emissions_tco2e_previous")
    scope2_previous = kpis.get("kpi_scope2_emissions_tco2e_previous")
    if "kpi_scope1_scope2_total_tco2e_current" not in kpis and scope1_current is not None and scope2_current is not None:
        kpis["kpi_scope1_scope2_total_tco2e_current"] = scope1_current + scope2_current
    if "kpi_scope1_scope2_total_tco2e_previous" not in kpis and scope1_previous is not None and scope2_previous is not None:
        kpis["kpi_scope1_scope2_total_tco2e_previous"] = scope1_previous + scope2_previous

    for _, (current_field, previous_field, yoy_field) in YOY_SPECS.items():
        calculated = compute_yoy(kpis.get(current_field), kpis.get(previous_field))
        if calculated is not None:
            kpis[yoy_field] = calculated
        elif yoy_field in kpis and (
            current_field not in kpis or previous_field not in kpis
        ):
            kpis.pop(yoy_field, None)

    return kpis


def row_quality(kpis: dict[str, Any], metadata: dict[str, str]) -> tuple[int, int]:
    numeric_count = sum(1 for key in kpis if key in NUMERIC_KPI_FIELDS)
    metadata_count = sum(1 for value in metadata.values() if value)
    return numeric_count, metadata_count


def load_source_rows(
    csv_paths: list[Path],
    company_map: dict[str, dict[str, str]],
) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for path in csv_paths:
        with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
            for raw_row in csv.DictReader(handle):
                raw_company = clean_text(raw_row.get("company"))
                mapped = company_map.get(normalize_source_key(raw_company), {})
                company = clean_text(mapped.get("company")) or title_company_name(raw_company)
                key = canonical_company_key(company)
                if not company or not key:
                    continue
                rows.append(
                    SourceRow(
                        row=raw_row,
                        source_file=str(path),
                        source_dataset=source_label(path),
                        company=company,
                        company_key=key,
                        nse_symbol=clean_text(mapped.get("nse_symbol")),
                    )
                )
    return rows


def build_metadata(source: SourceRow) -> dict[str, str]:
    row = source.row
    metadata = {
        "company": source.company,
        "reporting_year": normalize_reporting_year(row.get("reporting_year")),
        "sector": clean_text(row.get("meta_sector")),
        "market_cap": clean_text(row.get("meta_market_cap")),
        "framework": clean_text(row.get("meta_framework_used")),
        "brsr_version": clean_text(row.get("meta_brsr_version")),
        "assurance": clean_text(row.get("meta_assurance_type")),
        "geography": clean_text(row.get("meta_geography")),
        "nse_symbol": source.nse_symbol,
    }
    cleaned = {
        "company": source.company,
        "reporting_year": metadata["reporting_year"],
    }
    cleaned.update(
        {
            key: value
            for key, value in metadata.items()
            if key not in {"company", "reporting_year"}
            and value
            and not is_missing(value)
        }
    )
    return cleaned


def deduplicate(rows: list[SourceRow]) -> tuple[list[SourceRow], dict[str, Any]]:
    grouped: dict[str, list[SourceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.company_key].append(row)

    selected: list[SourceRow] = []
    duplicate_audit: list[dict[str, Any]] = []
    for company_key, candidates in grouped.items():
        ranked = sorted(
            candidates,
            key=lambda item: row_quality(clean_kpis(item.row), build_metadata(item)),
            reverse=True,
        )
        winner = ranked[0]
        display_name = max(
            (item.company for item in candidates),
            key=lambda name: (
                "limited" in name.lower(),
                " " in name,
                not name.isupper(),
                len(name),
            ),
        )
        winner.company = display_name
        selected.append(winner)
        if len(ranked) > 1:
            duplicate_audit.append(
                {
                    "company_key": company_key,
                    "selected_company": winner.company,
                    "selected_source": winner.source_file,
                    "candidate_count": len(ranked),
                    "discarded_sources": [item.source_file for item in ranked[1:]],
                }
            )
    return selected, {
        "source_rows": len(rows),
        "unique_company_keys": len(grouped),
        "duplicate_company_groups": len(duplicate_audit),
        "duplicate_rows_removed": len(rows) - len(selected),
        "duplicate_audit": duplicate_audit,
    }


def movement_sentence(
    metric: str,
    current: Any,
    previous: Any,
    yoy: Any,
    unit: str,
) -> tuple[str | None, list[dict[str, str]]]:
    current_text = format_number(current)
    previous_text = format_number(previous)
    if current_text is None:
        return None, []

    facts = [{"field": metric, "value": current_text, "unit": unit, "kind": "current"}]
    if previous_text is None:
        return f"{metric} stood at {current_text} {unit}.", facts

    facts.append({"field": metric, "value": previous_text, "unit": unit, "kind": "previous"})
    sentence = f"{metric} stood at {current_text} {unit}, compared with {previous_text} {unit} in the previous year."
    yoy_number = as_number(yoy)
    yoy_text = format_percent(yoy_number)
    if yoy_number is not None and yoy_text:
        direction = "reduced" if yoy_number > 0 else "increased" if yoy_number < 0 else "remained unchanged"
        if direction == "remained unchanged":
            sentence += f" {metric} remained unchanged year-on-year."
        else:
            sentence += f" {metric} {direction} by {yoy_text} year-on-year."
        facts.append(
            {
                "field": metric,
                "value": yoy_text,
                "unit": "%",
                "kind": "yoy",
                "direction": direction,
            }
        )
    return sentence, facts


def append_simple_fact(
    sentences: list[str],
    facts: list[dict[str, str]],
    label: str,
    value: Any,
    unit: str,
    sentence_template: str,
) -> None:
    formatted = format_number(value)
    if formatted is None:
        return
    sentences.append(sentence_template.format(value=formatted, unit=unit))
    facts.append({"field": label, "value": formatted, "unit": unit, "kind": "current"})


def build_target(metadata: dict[str, str], kpis: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    company = metadata["company"]
    year = metadata.get("reporting_year", "the reporting year")
    coverage = []
    if any(key.startswith("kpi_scope") for key in kpis):
        coverage.append("greenhouse gas emissions")
    if any("energy" in key for key in kpis):
        coverage.append("energy")
    if any("water" in key for key in kpis):
        coverage.append("water")
    if any("waste" in key for key in kpis):
        coverage.append("waste")
    if any(key in kpis for key in ("kpi_female_employee_percent", "kpi_women_on_board_percent")):
        coverage.append("workforce and board diversity")

    coverage_text = ", ".join(coverage[:-1]) + (" and " + coverage[-1] if len(coverage) > 1 else coverage[0] if coverage else "")
    first = (
        f"{company} disclosed selected ESG performance indicators for {year} "
        "based on the available structured BRSR KPI data."
    )
    if coverage_text:
        first += f" The available indicators cover {coverage_text}."
    paragraphs = [first]
    facts: list[dict[str, str]] = []

    emissions = []
    for label, prefix in (
        ("Scope 1 emissions", "scope1"),
        ("Scope 2 emissions", "scope2"),
        ("Scope 3 emissions", "scope3"),
        ("Combined Scope 1 and Scope 2 emissions", "scope1_scope2_total"),
    ):
        current_field, previous_field, yoy_field = YOY_SPECS[prefix]
        sentence, new_facts = movement_sentence(
            label,
            kpis.get(current_field),
            kpis.get(previous_field),
            kpis.get(yoy_field),
            "tCO2e",
        )
        if sentence:
            emissions.append(sentence)
            facts.extend(new_facts)
    if emissions:
        paragraphs.append(" ".join(emissions))

    resources: list[str] = []
    append_simple_fact(
        resources,
        facts,
        "Total energy consumption",
        kpis.get("kpi_total_energy_consumption_gj"),
        "GJ",
        "Total energy consumption was {value} {unit}.",
    )
    append_simple_fact(
        resources,
        facts,
        "Renewable energy consumption",
        kpis.get("kpi_renewable_energy_consumption_gj"),
        "GJ",
        "Renewable energy consumption was {value} {unit}.",
    )
    renewable_percent = format_percent(kpis.get("kpi_renewable_energy_percent"))
    if renewable_percent:
        resources.append(f"Renewable energy represented {renewable_percent} of the disclosed energy mix.")
        facts.append({"field": "Renewable energy share", "value": renewable_percent, "unit": "%", "kind": "current"})

    energy_intensity_unit = clean_text(kpis.get("kpi_energy_intensity_unit")) or "in the disclosed unit"
    sentence, new_facts = movement_sentence(
        "Energy intensity",
        kpis.get("kpi_energy_intensity_current"),
        kpis.get("kpi_energy_intensity_previous"),
        kpis.get("kpi_energy_intensity_yoy_reduction_percent"),
        energy_intensity_unit,
    )
    if sentence:
        resources.append(sentence)
        facts.extend(new_facts)

    for label, prefix in (
        ("Water consumption", "water_consumption"),
        ("Water withdrawal", "water_withdrawal"),
    ):
        current_field, previous_field, yoy_field = YOY_SPECS[prefix]
        sentence, new_facts = movement_sentence(
            label,
            kpis.get(current_field),
            kpis.get(previous_field),
            kpis.get(yoy_field),
            "KL",
        )
        if sentence:
            resources.append(sentence)
            facts.extend(new_facts)

    waste_unit = clean_text(kpis.get("kpi_waste_recycled_unit")) or "tonnes"
    total_waste_current = format_number(kpis.get("kpi_total_waste_generated_current"))
    total_waste_previous = format_number(kpis.get("kpi_total_waste_generated_previous"))
    if total_waste_current:
        text = f"Total waste generated was {total_waste_current} {waste_unit}"
        facts.append({"field": "Total waste generated", "value": total_waste_current, "unit": waste_unit, "kind": "current"})
        if total_waste_previous:
            text += f", compared with {total_waste_previous} {waste_unit} in the previous year"
            facts.append({"field": "Total waste generated", "value": total_waste_previous, "unit": waste_unit, "kind": "previous"})
        resources.append(text + ".")

    recycled_current = format_number(kpis.get("kpi_waste_recycled_current"))
    recycled_previous = format_number(kpis.get("kpi_waste_recycled_previous"))
    if recycled_current:
        text = f"Waste recycled or recovered was {recycled_current} {waste_unit}"
        facts.append({"field": "Waste recycled or recovered", "value": recycled_current, "unit": waste_unit, "kind": "current"})
        if recycled_previous:
            text += f", compared with {recycled_previous} {waste_unit} in the previous year"
            facts.append({"field": "Waste recycled or recovered", "value": recycled_previous, "unit": waste_unit, "kind": "previous"})
        resources.append(text + ".")

    recycled_percent = format_percent(kpis.get("kpi_waste_recycled_percent"))
    if recycled_percent:
        resources.append(f"The structured KPI data reports a waste recycling or recovery share of {recycled_percent}.")
        facts.append({"field": "Waste recycling or recovery share", "value": recycled_percent, "unit": "%", "kind": "current"})

    if resources:
        paragraphs.append(" ".join(resources))

    social: list[str] = []
    female_percent = format_percent(kpis.get("kpi_female_employee_percent"))
    if female_percent:
        social.append(f"Female employees represented {female_percent} of the disclosed workforce.")
        facts.append({"field": "Female employees", "value": female_percent, "unit": "%", "kind": "current"})
    board_percent = format_percent(kpis.get("kpi_women_on_board_percent"))
    if board_percent:
        social.append(f"Women represented {board_percent} of the disclosed Board composition.")
        facts.append({"field": "Women on the Board", "value": board_percent, "unit": "%", "kind": "current"})
    if social:
        paragraphs.append(" ".join(social))

    closing: list[str] = []
    net_zero_year = clean_text(kpis.get("kpi_net_zero_target_year"))
    if net_zero_year:
        closing.append(f"The structured KPI data records a net-zero target year of {net_zero_year}.")
        facts.append({"field": "Net-zero target year", "value": net_zero_year, "unit": "year", "kind": "target"})
    if metadata.get("assurance"):
        closing.append(f"The reported assurance basis is {metadata['assurance']}.")
    closing.append(
        "This summary is limited to the supplied structured KPIs and metadata and "
        "does not add unsupported policies, initiatives, awards, committees, "
        "certifications, or external-framework claims."
    )
    paragraphs.append(" ".join(closing))

    return "\n\n".join(paragraphs), facts


def build_prompt(metadata: dict[str, str], kpis: dict[str, Any]) -> str:
    metadata_lines = [
        f"- {key.replace('_', ' ').title()}: {value}"
        for key, value in metadata.items()
    ]
    kpi_lines = []
    for field in NUMERIC_KPI_FIELDS + TEXT_KPI_FIELDS:
        if field not in kpis:
            continue
        value = format_number(kpis[field]) if field in NUMERIC_KPI_FIELDS else clean_text(kpis[field])
        if value is not None:
            kpi_lines.append(f"- {FIELD_LABELS[field]}: {value}")
    return (
        "Task:\n"
        "Write one professional ESG narrative summary that can be used as the "
        "starting point for a company report. Cover every supplied KPI, compare "
        "current and previous values where available, and do not add unsupported claims.\n\n"
        "Company metadata:\n"
        + "\n".join(metadata_lines)
        + "\n\nKPI data:\n"
        + ("\n".join(kpi_lines) if kpi_lines else "No valid structured KPI values supplied.")
    )


def stable_split(company_key: str) -> str:
    bucket = int(hashlib.sha256(company_key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def make_example(source: SourceRow) -> dict[str, Any] | None:
    metadata = build_metadata(source)
    if metadata.get("reporting_year") not in {"FY 2024-25", "FY 2025-26"}:
        return None
    kpis = clean_kpis(source.row)
    if len(kpis) < 2:
        return None
    target, expected_facts = build_target(metadata, kpis)
    if not expected_facts:
        return None
    prompt = build_prompt(metadata, kpis)
    split = stable_split(source.company_key)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "prompt": prompt,
        "target_summary": target,
        "company": metadata["company"],
        "company_key": source.company_key,
        "reporting_year": metadata.get("reporting_year"),
        "source_dataset": source.source_dataset,
        "source_file": source.source_file,
        "split": split,
        "metadata": metadata,
        "kpis": kpis,
        "expected_facts": expected_facts,
        "num_kpis": len(kpis),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_quality_csv(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "company",
        "company_key",
        "split",
        "reporting_year",
        "source_dataset",
        "num_kpis",
        "num_expected_facts",
        "prompt_chars",
        "target_chars",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "company": example["company"],
                    "company_key": example["company_key"],
                    "split": example["split"],
                    "reporting_year": example["reporting_year"],
                    "source_dataset": example["source_dataset"],
                    "num_kpis": example["num_kpis"],
                    "num_expected_facts": len(example["expected_facts"]),
                    "prompt_chars": len(example["prompt"]),
                    "target_chars": len(example["target_summary"]),
                }
            )


def build_dataset(
    csv_paths: list[Path],
    output_dir: Path,
    company_map_path: Path | None = None,
) -> dict[str, Any]:
    company_map = load_company_map(company_map_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if company_map:
        with (output_dir / "company_name_map.json").open("w", encoding="utf-8") as handle:
            json.dump(company_map, handle, indent=2, ensure_ascii=False)
    source_rows = load_source_rows(csv_paths, company_map)
    selected_rows, dedupe_manifest = deduplicate(source_rows)
    examples = [example for row in selected_rows if (example := make_example(row))]
    examples.sort(key=lambda item: (item["split"], item["company_key"]))

    split_rows = {
        split: [row for row in examples if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    write_jsonl(output_dir / "kpi_summary_all.jsonl", examples)
    for split, rows in split_rows.items():
        write_jsonl(output_dir / f"kpi_summary_{split}.jsonl", rows)
    write_quality_csv(output_dir / "dataset_quality.csv", examples)

    company_sets = {
        split: {row["company_key"] for row in rows}
        for split, rows in split_rows.items()
    }
    overlap = {
        "train_validation": sorted(company_sets["train"] & company_sets["validation"]),
        "train_test": sorted(company_sets["train"] & company_sets["test"]),
        "validation_test": sorted(company_sets["validation"] & company_sets["test"]),
    }
    manifest = {
        "task": "company_metadata_and_kpis_to_grounded_esg_summary",
        "source_files": [str(path) for path in csv_paths],
        "company_map_source": company_map_path.name if company_map_path else None,
        "portable_company_map": "company_name_map.json" if company_map else None,
        "excluded_inputs": [
            "llm_training_summary",
            "intent labels and counts",
            "section classification flags and scores",
            "KPI evidence text",
            "SusGen",
            "annual reports 2022-23",
        ],
        "source_rows": dedupe_manifest["source_rows"],
        "deduplicated_company_rows": len(selected_rows),
        "examples_after_quality_filter": len(examples),
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "company_overlap": overlap,
        "kpi_count_distribution": dict(Counter(row["num_kpis"] for row in examples)),
        "source_distribution": dict(Counter(row["source_dataset"] for row in examples)),
        "reporting_year_distribution": dict(Counter(row["reporting_year"] for row in examples)),
        "deduplication": dedupe_manifest,
    }
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    sample_rows = []
    for split in ("train", "validation", "test"):
        sample_rows.extend(split_rows[split][:1])
    with (output_dir / "sample_summaries.json").open("w", encoding="utf-8") as handle:
        json.dump(sample_rows, handle, indent=2, ensure_ascii=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        help="The BRSR 2024-25 and 2025-26 PRD master CSV files or their directory.",
    )
    parser.add_argument(
        "--company-map-placeholder-jsonl",
        type=Path,
        default=None,
        help="Optional placeholder JSONL used only to recover clean company names for raw 2025-26 filenames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/kpi_summary"),
    )
    args = parser.parse_args()

    csv_paths = find_csvs(args.paths)
    if not csv_paths:
        raise FileNotFoundError("No BRSR 2024-25 or 2025-26 PRD master CSV files were found.")
    print("Using source files:")
    for path in csv_paths:
        print("-", path)
    manifest = build_dataset(
        csv_paths,
        args.output_dir,
        args.company_map_placeholder_jsonl,
    )
    print(json.dumps(
        {
            "examples": manifest["examples_after_quality_filter"],
            "splits": manifest["split_counts"],
            "overlap": manifest["company_overlap"],
            "output_dir": str(args.output_dir),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
