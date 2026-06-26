# CarbonTatvaAI ESG Intelligence

CarbonTatvaAI is an ESG reporting and benchmarking workspace for Indian BRSR disclosures.

The current repo focuses on three tracks:

1. **BRSR corpus collection**: scrape NSE BRSR PDFs and export Kaggle-style year folders/zips.
2. **KPI-to-ESG summary fine-tuning**: prepare KPI-to-summary datasets and fine-tune Llama 3.1 8B with Unsloth QLoRA.
3. **ESG benchmarking engine**: compare a company's KPIs/disclosures against peer historical BRSR data to identify missing KPIs, weak disclosures, and sector adoption trends.

## Repository Layout

```text
artifacts/
  benchmark_engine/        Small generated benchmark CSV/JSON/Markdown samples
configs/
  Modelfile.template       Ollama adapter Modelfile template
docs/
  *_RUNBOOK.md             How to run each stage
notebooks/
  *.ipynb                  Kaggle/Colab notebooks used for fine-tuning and chatbot work
scripts/
  scraping/                NSE BRSR and annual-report scrapers
  benchmarking/            Benchmark database and demo report builders
  training/                Dataset preparation, fine-tuning, evaluation scripts
  deployment/              Ollama/RAG helper scripts
  legacy/                  Earlier exploration scripts kept for reference
tests/
  test_build_kpi_summary_dataset.py
```

## What Is Not Stored In Git

Large raw report PDFs, year-wise BRSR zips, model checkpoints, LoRA adapters, and local datasets are intentionally excluded.

GitHub should hold the code and small reproducible artifacts. The generated BRSR PDF corpus should live in Drive/Kaggle/object storage.

## BRSR Scraping

Generate missing newer BRSR report folders in the same style as the existing Kaggle dataset:

```powershell
python scripts/scraping/scrape_nse_brsr_reports.py --years 2025-26,2024-25 --download
```

Output shape:

```text
data/scraped_brsr_reports/
  kaggle_format/
    2024_2025/
      3M India Limited.pdf
    2025_2026/
      Reliance Industries Limited.pdf
  zips/
    brsr_reports_2024_2025.zip
    brsr_reports_2025_2026.zip
```

The scraper uses NSE's public BRSR endpoint:

```text
https://www.nseindia.com/api/corporate-bussiness-sustainabilitiy
```

NSE spells it as `bussiness-sustainabilitiy`.

## Benchmarking Demo

The benchmark artifacts in `artifacts/benchmark_engine/` are small generated samples showing:

- source inventory
- company-year benchmark rows
- KPI availability by sector/year
- disclosure adoption by sector/year
- one sample benchmark report

Run the demo script:

```powershell
python scripts/benchmarking/benchmark_engine_demo.py
```

## Comparative Benchmarking

Sector-wise comparison:

```powershell
python scripts/benchmarking/comparative_benchmark.py --company "360 ONE WAM" --year "FY 2024-25"
```

Custom peer-group comparison:

```powershell
python scripts/benchmarking/comparative_benchmark.py --company "TCS" --year "FY 2024-25" --peers "INFOSYS,WIPRO,HCL,TECH MAHINDRA"
```

The output includes:

- company value
- peer average and median
- peer min/max range
- rank within selected peer group
- missing KPI opportunities
- high-adoption disclosure gaps

## KPI Summary Fine-Tuning

Primary notebook:

```text
notebooks/CarbonTatvaAI_KPI_Summary_Unsloth.ipynb
```

Presentation statement:

> I prepared a company-level BRSR dataset where metadata and structured KPIs are the input and a grounded ESG narrative summary is the output. I fine-tuned Llama 3.1 8B using Unsloth QLoRA and evaluated numerical fidelity and KPI coverage on unseen companies.

## Current Product Direction

Current assigned task:

> ESG Intelligence & Benchmarking Engine: benchmark disclosures against industry leaders, identify gaps, and strengthen ESG reporting before publication.

Practical V1 flow:

```text
BRSR/annual report PDFs or KPI data
    -> extraction and taxonomy mapping
    -> benchmark database
    -> peer comparison
    -> gap analysis
    -> evidence-backed recommendations
```

## Quick Validation

```powershell
python -m pytest tests
```

If `pytest` is unavailable:

```powershell
python -m unittest discover -s tests
```
