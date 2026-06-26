# ESG Intelligence & Benchmarking Engine Starter

## Current Task

New track:

**ESG Intelligence & Benchmarking Engine - Snehashish and Pratyush**

Tagline:

> Benchmark your disclosures against industry leaders, identify gaps, and strengthen ESG reporting before publication.

## What We Have Locally

The first step is to build the benchmark corpus using existing data:

- Annual Reports 2022-23 extracted KPI file.
- Annual Reports 2022-23 paragraph-intent file.
- PRD/BRSR master datasets for 2022-23, 2024-25, and 2025-26.
- KPI summary train/test/validation files from the previous KPI-to-summary track.

Incomplete `.part` files are excluded.

## How To Build Starter Benchmark Data

Run:

```powershell
python prepare_benchmark_engine_data.py
```

Fast inventory-only run:

```powershell
python prepare_benchmark_engine_data.py --skip-heavy-intents
```

## Output Folder

`data/benchmark_engine/`

Key files:

- `source_inventory.csv`
- `benchmark_company_year.csv`
- `disclosure_adoption_by_sector_year.csv`
- `kpi_availability_by_sector_year.csv`
- `annual_report_intent_counts.csv`
- `annual_report_section_counts.csv`
- `annual_report_intent_samples.csv`
- `benchmark_engine_manifest.json`

## What This Enables

The outputs are the starting point for:

- peer universe selection by sector/year,
- disclosure coverage benchmarking,
- KPI gap detection,
- emerging trend analysis,
- evidence-backed recommendation generation,
- dashboard cards for executive summary, peer benchmark, and competitor insights.

## Next Implementation Step

Build the V1 benchmarking function:

Input:

- target company/year,
- selected benchmark sector or custom peer list,
- connected local BRSR/annual-report corpus and KPI dataset.

Output:

- missing disclosures,
- weak disclosures,
- KPI gaps,
- sector adoption percentages,
- examples from peers,
- ranked recommendations.
