# CarbonTatvaAI Internship Work Overview

Prepared by: Pratyush  
Project: CarbonTatvaAI ESG Intelligence and Benchmarking Engine  
Date: 27 July 2026

## 1. Project Overview

- Worked on CarbonTatvaAI, an ESG intelligence and benchmarking project.
- The project focuses on Indian ESG/BRSR disclosures.
- The goal was to help compare a company's ESG data with sector peers or custom peer groups.
- The system is meant to identify:
  - KPI gaps
  - disclosure gaps
  - peer comparison insights
  - report evidence coverage

## 2. Main Final Output

- Built an ESG benchmarking dashboard.
- The dashboard lets a user:
  - select a company
  - select a reporting year
  - compare with a sector
  - compare with a custom group of companies
  - view KPI benchmark values
  - view missing KPI opportunities
  - view disclosure gaps
  - view connected report evidence

## 3. Main Links

- GitHub repository:
  - https://github.com/propixx/CarbonTatvaAI-ESG-Intelligence

- GitHub repository ZIP:
  - https://github.com/propixx/CarbonTatvaAI-ESG-Intelligence/archive/refs/heads/main.zip

- Deployed dashboard:
  - https://propixx.github.io/CarbonTatvaAI-ESG-Intelligence/dashboard/

- Literature review:
  - https://docs.google.com/spreadsheets/d/1DLYi0VMLkZ1fcprrTtyAzilIK01IgsiuU4mhVYl1zBM/edit?usp=sharing

- PRD datasets:
  - https://drive.google.com/drive/folders/1HgnCVge9d9-EWDv9QbOiRMrFRylmPge5?usp=drive_link

- Additional report/data drive:
  - https://drive.google.com/drive/folders/18xfBj5-S4erVfAhyFHz7PSSZZtWjRhvg?usp=drive_link

- Stitch UI reference:
  - https://stitch.withgoogle.com/projects/15316556009456935239

## 4. Literature Review Work

- Reviewed ESG and LLM-related papers.
- Categorized papers by:
  - end use case
  - methodology
  - dataset used
  - advantages
  - limitations
  - project relevance

- Main research areas studied:
  - ESG report generation
  - ESG question answering
  - sustainability report summarization
  - RAG over ESG reports
  - ESG-specific language models
  - ESG classification
  - disclosure benchmarking
  - hallucination reduction

- Important papers studied:
  - SusGen-GPT
  - ESGReveal
  - ESGBERT
  - ESG-CID
  - ESG-Bench
  - other ESG/finance NLP papers

## 5. Key Research Learning

- ESG systems should not depend only on free-form LLM generation.
- ESG outputs need strong grounding because numbers and claims must be reliable.
- For benchmarking, statistics should be calculated using code, not guessed by an LLM.
- LLMs are better suited for:
  - explanation
  - report wording
  - summarization
  - evidence-based recommendations

- Final approach:
  - use structured data for benchmark calculations
  - use report evidence for grounding
  - use LLM/RAG later for natural-language explanation

## 6. Data Work

- Worked with structured ESG/BRSR datasets.
- Used PRD master datasets as the main benchmark data source.
- Indexed BRSR reports from available local and zipped datasets.
- Connected annual-report extracted evidence where raw PDFs were too large.

- Main data sources:
  - PRD/BRSR master datasets
  - BRSR 2021-24 report zip
  - BRSR 2024-25 scraped report data
  - BRSR 2025-26 scraped report data
  - annual-report KPI and paragraph evidence
  - SusGen-30K for LLM fine-tuning exploration

## 7. Connected Data Summary

- Total report/evidence rows indexed:
  - 5,813

- BRSR report entries:
  - 4,411

- Annual-report extracted evidence entries:
  - 1,402

- Benchmark company-year rows linked with evidence:
  - 3,864

- BRSR report entries by year:
  - FY 2021-22: 177
  - FY 2022-23: 801
  - FY 2023-24: 1,059
  - FY 2024-25: 2,182
  - FY 2025-26: 192

## 8. Fine-Tuning Exploration

- Explored fine-tuning for KPI-to-ESG narrative generation.
- Worked with:
  - Llama 3.1 8B Instruct
  - Unsloth
  - QLoRA
  - LoRA adapters
  - Kaggle/Colab training flow

- Training task explored:
  - input: company metadata and KPI values
  - output: ESG narrative summary

- Added logic for:
  - response-only training
  - checkpoint saving
  - resume from checkpoint
  - evaluation planning

- This work was useful for report-generation experiments.
- Final product focus moved toward benchmarking because that matched the PRD better.

## 9. Benchmarking Engine Work

- Built backend logic for ESG benchmarking.
- The benchmarking engine supports:
  - company vs sector comparison
  - company vs custom peer group comparison
  - KPI comparison
  - average calculation
  - median calculation
  - range calculation
  - rank calculation
  - missing KPI detection
  - disclosure gap detection

- This was the main product logic.

## 10. Dashboard Work

- Built a static dashboard for the benchmark engine.
- The dashboard is hosted using GitHub Pages.
- It does not need a live backend for the demo.
- It uses exported benchmark data from the repo.

- Dashboard features:
  - company selector
  - year selector
  - sector benchmark mode
  - custom peer group mode
  - KPI comparison section
  - missing KPI section
  - disclosure gap section
  - report source coverage section
  - visual comparison section

## 11. Scraping Work

- Added scripts for report collection.
- Worked on:
  - BRSR report scraping
  - annual report scraping
  - year-wise folder output
  - Kaggle-style ZIP formatting

- Main scraping scripts:
  - `scripts/scraping/scrape_nse_brsr_reports.py`
  - `scripts/scraping/scrape_nse_annual_reports.py`

## 12. Important Code Files

- Benchmarking:
  - `scripts/benchmarking/prepare_benchmark_engine_data.py`
  - `scripts/benchmarking/comparative_benchmark.py`
  - `scripts/benchmarking/connect_report_corpus.py`
  - `scripts/benchmarking/export_dashboard_data.py`
  - `scripts/benchmarking/benchmark_engine_demo.py`

- Dashboard:
  - `dashboard/index.html`
  - `dashboard/styles.css`
  - `dashboard/app.js`
  - `dashboard/dashboard_data.js`

- Training:
  - `scripts/training/build_kpi_summary_dataset.py`
  - `scripts/training/build_sectionwise_training_data.py`
  - `scripts/training/train_carbontatva.py`
  - `scripts/training/evaluate_model.py`

- Deployment/RAG:
  - `scripts/deployment/ollama_kpi_client.py`
  - `scripts/deployment/rag_inference.py`

- Notebooks:
  - `notebooks/CarbonTatvaAI_ESG_Benchmark_Demo.ipynb`
  - `notebooks/CarbonTatvaAI_KPI_Summary_Unsloth.ipynb`
  - `notebooks/CarbonTatvaAI_Enterprise_ESG_Chatbot_Final.ipynb`

## 13. Important Documentation Files

- `README.md`
- `FINAL_INTERNSHIP_SUBMISSION_REPORT.md`
- `FINAL_INTERNSHIP_SIMPLE_OVERVIEW.md`
- `docs/CARBONTATVAI_BENCHMARKING_EXPLAINER.md`
- `docs/INTERVIEW_PROJECT_STORY.md`
- `docs/RESEARCH_FLEX_NOTES.md`
- `docs/BRSR_REPORT_SCRAPING_RUNBOOK.md`
- `docs/ANNUAL_REPORT_SCRAPING_RUNBOOK.md`
- `docs/KPI_SUMMARY_RUNBOOK.md`
- `docs/ENTERPRISE_CHATBOT_RUNBOOK.md`

## 14. Testing

- Ran repository tests.
- Test command:
  - `python -m pytest tests`

- Result:
  - 4 tests passed

## 15. Current Status

- Completed:
  - literature review
  - data organization
  - BRSR/report indexing
  - benchmark engine scripts
  - custom peer group comparison
  - sector comparison
  - dashboard
  - GitHub Pages deployment
  - documentation
  - submission report

- Partially explored:
  - Llama fine-tuning
  - Unsloth QLoRA
  - KPI-to-summary generation
  - RAG and Ollama deployment

- Future work:
  - full production backend
  - database integration
  - user authentication
  - live LLM/RAG explanation layer
  - PDF evidence snippet extraction
  - standards mapping to BRSR/GRI/ESRS
  - downloadable benchmark reports

## 16. What To Submit

- Submit the final ZIP:
  - `CarbonTatvaAI_Final_Submission_Bundle.zip`

- Submit the GitHub repo:
  - https://github.com/propixx/CarbonTatvaAI-ESG-Intelligence

- Submit the deployed dashboard:
  - https://propixx.github.io/CarbonTatvaAI-ESG-Intelligence/dashboard/

- Submit this overview file:
  - `FINAL_INTERNSHIP_SIMPLE_OVERVIEW.md`

## 17. Short Final Description

- I worked on ESG intelligence and benchmarking for CarbonTatvaAI.
- I reviewed ESG + LLM research, prepared and connected ESG/BRSR datasets, explored KPI-to-summary fine-tuning, and built a real-data benchmark dashboard.
- The final dashboard compares a company against sector or custom peers and shows KPI benchmarks, missing KPIs, disclosure gaps, and evidence coverage.

