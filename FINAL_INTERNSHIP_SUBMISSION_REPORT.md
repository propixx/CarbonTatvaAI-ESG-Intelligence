# CarbonTatvaAI Internship Final Submission Report

Prepared by: Pratyush  
Project: CarbonTatvaAI ESG Intelligence and Benchmarking Engine  
Last updated: 27 July 2026  

## 1. Executive Summary

During the internship, I worked on the ESG intelligence layer for CarbonTatvaAI. The work started from research and experimentation around ESG-focused LLMs, sustainability report generation, RAG, and fine-tuning, and then moved toward the product direction requested in the PRD: an ESG benchmarking engine.

The final direction was not just to build a chatbot. The core product idea became:

```text
Company ESG/BRSR data
    -> compare against sector or custom peer group
    -> calculate benchmark statistics
    -> identify missing KPIs and disclosure gaps
    -> support report improvement before publication
```

The main delivered output is a working static ESG benchmarking dashboard backed by real indexed BRSR/PRD data. It supports company-year selection, sector-wise comparison, custom peer-group comparison, KPI benchmarking, missing KPI opportunities, disclosure gap detection, and report-source coverage.

## 2. Main Submission Links

| Item | Link / Path |
|---|---|
| GitHub repository | https://github.com/propixx/CarbonTatvaAI-ESG-Intelligence |
| GitHub repository ZIP | https://github.com/propixx/CarbonTatvaAI-ESG-Intelligence/archive/refs/heads/main.zip |
| Deployed dashboard | https://propixx.github.io/CarbonTatvaAI-ESG-Intelligence/dashboard/ |
| Local dashboard share ZIP | `CarbonTatvaAI_ESG_Benchmark_Dashboard_Share.zip` |
| Literature review sheet | https://docs.google.com/spreadsheets/d/1DLYi0VMLkZ1fcprrTtyAzilIK01IgsiuU4mhVYl1zBM/edit?usp=sharing |
| PRD datasets Drive | https://drive.google.com/drive/folders/1HgnCVge9d9-EWDv9QbOiRMrFRylmPge5?usp=drive_link |
| Additional report/data Drive | https://drive.google.com/drive/folders/18xfBj5-S4erVfAhyFHz7PSSZZtWjRhvg?usp=drive_link |
| Stitch UI reference/prototype | https://stitch.withgoogle.com/projects/15316556009456935239 |

Note: Large PDFs, model checkpoints, and LoRA adapter weights are intentionally not stored in GitHub because they are large artifacts. The GitHub repo contains code, notebooks, documentation, and small reproducible benchmark artifacts.

## 3. Final Project Statement

The project can be described as:

> I worked on an ESG Intelligence and Benchmarking Engine for Indian BRSR disclosures. I connected structured PRD/BRSR KPI data with indexed BRSR and annual-report evidence, then built a dashboard that compares a selected company against a sector or custom peer group. The system shows KPI values, peer average, median, range, rank, missing KPI opportunities, disclosure gaps, and evidence coverage.

Short version:

> I built a real-data ESG benchmarking dashboard that helps companies understand how their ESG disclosures compare with peers before publication.

## 4. What I Worked On

### 4.1 Literature Review and Research Direction

I reviewed ESG + LLM papers and organized them by:

- end use case
- summary
- methodology
- data used
- link
- pros
- cons

The major research themes identified were:

- ESG report generation
- ESG question answering
- RAG over sustainability reports
- ESG-specific language models
- ESG classification and sentiment analysis
- disclosure mapping to standards such as GRI, ESRS, and BRSR
- hallucination reduction and evidence-grounded answers

Important papers and how they influenced the work:

| Paper / Direction | What It Helped Decide |
|---|---|
| SusGen-GPT | ESG generation needs domain-specific data and report grounding, not just generic prompting. |
| ESGReveal | ESG systems should extract and structure report evidence before generating answers. |
| ESGBERT | ESG text has domain-specific meaning, so generic NLP models are often not enough. |
| ESG-CID | Future systems should map company disclosures to reporting standards and retrieve evidence. |
| ESG-Bench / hallucination work | ESG outputs should be grounded and verified because hallucinated numbers are risky. |

Main research-backed design decision:

> The LLM should explain the evidence, not become the evidence. For benchmarking, numerical calculations such as median, average, rank, and peer range should be deterministic.

### 4.2 Initial LLM and Fine-Tuning Exploration

The earlier phase explored:

- Llama 3.1 8B Instruct
- SusGen-30K
- QLoRA
- Unsloth fine-tuning
- KPI-to-ESG narrative generation
- response-only supervised fine-tuning
- 15-minute checkpointing for long Kaggle/Colab runs
- evaluation ideas such as ROUGE-L, BERTScore, KPI coverage, and numerical fidelity

This stage helped clarify the difference between:

- fine-tuning for ESG writing style
- RAG for report-grounded answers
- deterministic benchmarking for peer comparison

The final understanding was:

```text
Fine-tuning = useful for consistent ESG narrative generation
RAG = useful for retrieving evidence from reports
Benchmarking = useful for reliable peer comparison and gap detection
```

### 4.3 KPI-to-ESG Summary Dataset Preparation

I worked on preparing training data where:

```text
Input:
Company metadata + available BRSR KPI values

Output:
Grounded ESG narrative summary
```

The dataset preparation logic focused on:

- using structured KPI fields
- normalizing company names
- retaining company-year records
- calculating year-on-year movement before training
- avoiding unsupported claims
- creating clean train/validation/test splits
- keeping test companies separate from training companies where possible

Relevant notebook and scripts:

```text
notebooks/CarbonTatvaAI_KPI_Summary_Unsloth.ipynb
scripts/training/build_kpi_summary_dataset.py
scripts/training/build_sectionwise_training_data.py
```

This was useful for the report-generation side, but the final PRD direction prioritized benchmarking.

### 4.4 Report and Dataset Collection

I worked with available BRSR, PRD, and annual-report datasets.

Data sources connected or considered:

| Source | Use |
|---|---|
| PRD/BRSR master datasets | Main structured benchmark dataset |
| BRSR 2021-24 zip | Historical BRSR report corpus |
| Scraped BRSR 2024-25 and 2025-26 reports | Newer BRSR report coverage |
| Annual report extracted KPI/paragraph evidence | Annual-report evidence without carrying huge raw PDFs |
| SusGen-30K | LLM/fine-tuning exploration, not primary benchmark data |

Final connected corpus stats:

| Metric | Count |
|---|---:|
| Total report/evidence rows indexed | 5,813 |
| BRSR report entries | 4,411 |
| Annual-report extracted evidence entries | 1,402 |
| Benchmark company-year rows linked to evidence | 3,864 |

BRSR coverage by year:

| Reporting year | Indexed BRSR report entries |
|---|---:|
| FY 2021-22 | 177 |
| FY 2022-23 | 801 |
| FY 2023-24 | 1,059 |
| FY 2024-25 | 2,182 |
| FY 2025-26 | 192 |

Annual-report evidence:

| Reporting year | Indexed extracted evidence rows |
|---|---:|
| FY 2022-23 | 1,400 |

Important practical decision:

> Raw annual-report PDFs are large, so the current implementation uses extracted annual KPI and paragraph evidence. The connector is written so raw annual PDFs can be indexed later if they are added.

### 4.5 BRSR and Annual Report Scraping Scripts

I organized scraping code for:

- NSE BRSR report collection
- NSE annual report collection
- year-wise output folders
- Kaggle-style folder layout
- ZIP export for sharing

Relevant scripts:

```text
scripts/scraping/scrape_nse_brsr_reports.py
scripts/scraping/scrape_nse_annual_reports.py
```

The BRSR scraper is intended to produce a structure like:

```text
data/scraped_brsr_reports/
  kaggle_format/
    2024_2025/
      Company Name.pdf
    2025_2026/
      Company Name.pdf
  zips/
    brsr_reports_2024_2025.zip
    brsr_reports_2025_2026.zip
```

This matches the required year-wise dataset packaging style.

### 4.6 Benchmark Engine

The benchmark engine is the main final deliverable.

It supports:

- target company selection
- reporting year selection
- sector-wise comparison
- custom peer-group comparison
- KPI value comparison
- peer average
- peer median
- peer min/max range
- company rank
- missing KPI opportunities
- disclosure gap detection
- report-source coverage

Core scripts:

```text
scripts/benchmarking/prepare_benchmark_engine_data.py
scripts/benchmarking/comparative_benchmark.py
scripts/benchmarking/connect_report_corpus.py
scripts/benchmarking/export_dashboard_data.py
scripts/benchmarking/benchmark_engine_demo.py
```

Generated benchmark artifacts:

```text
artifacts/benchmark_engine/source_inventory.csv
artifacts/benchmark_engine/benchmark_company_year.csv
artifacts/benchmark_engine/disclosure_adoption_by_sector_year.csv
artifacts/benchmark_engine/kpi_availability_by_sector_year.csv
artifacts/benchmark_engine/report_corpus_index.csv
artifacts/benchmark_engine/benchmark_company_year_report_links.csv
artifacts/benchmark_engine/report_corpus_manifest.json
```

### 4.7 Static Dashboard

The dashboard is a production-shareable static frontend.

Dashboard files:

```text
dashboard/index.html
dashboard/styles.css
dashboard/app.js
dashboard/dashboard_data.js
```

The dashboard supports:

- company dropdown
- year dropdown
- benchmark mode selection
- sector peer comparison
- custom peer selection
- KPI benchmark table
- gap analysis
- report-source coverage
- visual summary tab

Deployment:

```text
https://propixx.github.io/CarbonTatvaAI-ESG-Intelligence/dashboard/
```

Why this is useful:

> The dashboard does not require a Python backend for demo purposes. It runs from exported benchmark data, so it can be hosted on GitHub Pages and shared easily.

## 5. Current Repository Layout

```text
CarbonTatvaAI-ESG-Intelligence/
  artifacts/
    benchmark_engine/
      small generated CSV/JSON/Markdown benchmark artifacts
  configs/
    Modelfile.template
  dashboard/
    static dashboard files
  docs/
    runbooks, explanations, research notes, interview notes
  notebooks/
    Kaggle/Colab notebooks for fine-tuning and demos
  scripts/
    benchmarking/
    deployment/
    legacy/
    scraping/
    training/
  tests/
    dataset-preparation tests
  README.md
  FINAL_INTERNSHIP_SUBMISSION_REPORT.md
```

## 6. Important Files To Mention

### Documentation

| File | Purpose |
|---|---|
| `README.md` | Main repo overview and run instructions |
| `FINAL_INTERNSHIP_SUBMISSION_REPORT.md` | End-of-internship report |
| `docs/CARBONTATVAI_BENCHMARKING_EXPLAINER.md` | Simple explanation of the benchmark engine |
| `docs/INTERVIEW_PROJECT_STORY.md` | Interview-style story of the project |
| `docs/RESEARCH_FLEX_NOTES.md` | Research-backed talking points |
| `docs/BRSR_REPORT_SCRAPING_RUNBOOK.md` | BRSR scraping instructions |
| `docs/ANNUAL_REPORT_SCRAPING_RUNBOOK.md` | Annual report scraping instructions |
| `docs/KPI_SUMMARY_RUNBOOK.md` | KPI-to-summary fine-tuning notes |
| `docs/ENTERPRISE_CHATBOT_RUNBOOK.md` | Chatbot/RAG deployment notes |

### Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/CarbonTatvaAI_ESG_Benchmark_Demo.ipynb` | Demo notebook for benchmarking |
| `notebooks/CarbonTatvaAI_KPI_Summary_Unsloth.ipynb` | Unsloth QLoRA KPI-to-summary fine-tuning |
| `notebooks/CarbonTatvaAI_Enterprise_ESG_Chatbot_Final.ipynb` | Enterprise chatbot/RAG exploration |
| `notebooks/CarbonTatvaAI_PRD_Master_Unsloth.ipynb` | PRD master fine-tuning exploration |
| `notebooks/CarbonTatvaAI_Unsloth_KPI_to_Report.ipynb` | KPI-to-report generation exploration |

### Scripts

| Script | Purpose |
|---|---|
| `scripts/benchmarking/comparative_benchmark.py` | CLI sector/custom peer benchmark generation |
| `scripts/benchmarking/connect_report_corpus.py` | Connect BRSR/annual evidence to company-year rows |
| `scripts/benchmarking/export_dashboard_data.py` | Export dashboard-ready JavaScript data |
| `scripts/benchmarking/prepare_benchmark_engine_data.py` | Prepare benchmark source tables |
| `scripts/scraping/scrape_nse_brsr_reports.py` | Scrape BRSR reports from NSE |
| `scripts/scraping/scrape_nse_annual_reports.py` | Scrape annual reports from NSE |
| `scripts/training/build_kpi_summary_dataset.py` | Prepare KPI-to-summary dataset |
| `scripts/deployment/ollama_kpi_client.py` | Local Ollama adapter client helper |
| `scripts/deployment/rag_inference.py` | RAG inference helper |

## 7. How The Final Dashboard Works

The dashboard flow is:

```text
User selects target company and year
        |
        v
User chooses sector benchmark or custom peer group
        |
        v
Dashboard calculates peer statistics
        |
        v
Dashboard shows KPI comparisons, ranks, gaps, and evidence coverage
```

For each KPI, the dashboard can show:

- selected company value
- peer median
- peer average
- peer min/max range
- company rank
- interpretation

For disclosure gaps, the dashboard checks:

- which ESG topics peers commonly disclose
- whether the target company is missing those topics
- whether the missing topic is a high-adoption peer standard

For report-source coverage, the dashboard shows:

- BRSR evidence count
- annual evidence count
- total connected evidence sources
- peer evidence availability

## 8. Why This Approach Is Correct For The PRD

The PRD is about ESG intelligence and benchmarking, not only free-form report generation.

The dashboard addresses the PRD because it:

- compares companies against industry peers
- supports both sector and custom peer groups
- identifies missing KPIs
- identifies disclosure gaps
- shows benchmark statistics
- connects structured data with report evidence
- gives a base for future chatbot or LLM explanation

The correct mental model is:

```text
Benchmark engine first.
LLM explanation second.
Full chatbot/RAG later.
```

This is important because ESG numbers should be calculated, not guessed by a model.

## 9. LLM, RAG, And Fine-Tuning Work

The internship also included LLM-side exploration.

### Fine-Tuning

Purpose:

```text
Teach a model to convert KPI inputs into ESG-style narrative summaries.
```

Model direction:

```text
Llama 3.1 8B Instruct + Unsloth QLoRA
```

Why QLoRA/Unsloth:

- lower GPU memory requirement
- faster fine-tuning on Kaggle/Colab GPUs
- LoRA adapter output instead of full model storage
- easier local deployment through Ollama-style adapter workflow

### RAG

Purpose:

```text
Retrieve relevant report evidence at query time and pass it to an LLM.
```

Why RAG is useful:

- company reports change every year
- avoids retraining for every new report
- improves factual grounding
- allows answers to cite report evidence

### Why LLM Is Not The Whole Product

For benchmarking:

```text
median, average, rank, range, missing KPI rate = deterministic code
```

For explanation:

```text
summaries, recommendations, report wording = LLM/RAG layer
```

This separation is the main technical design decision.

## 10. Validation And Testing

Validation was planned and partially implemented around:

- KPI numerical fidelity
- YoY increase/reduction correctness
- KPI coverage in generated summaries
- unsupported-number rate
- unsupported-claim rate
- ROUGE-L and BERTScore as secondary metrics
- held-out company examples

For repository checks:

```powershell
python -m pytest tests
```

If `pytest` is not available:

```powershell
python -m unittest discover -s tests
```

For dashboard verification:

```powershell
python -m http.server 8899 --directory dashboard
```

Then open:

```text
http://127.0.0.1:8899
```

## 11. What Is Production-Ready

Ready for demo/share:

- GitHub repo with code and documentation
- GitHub Pages hosted dashboard
- static dashboard folder
- benchmark artifacts
- dashboard share ZIP
- scraping scripts
- benchmark engine scripts
- runbooks and explanation docs

Not production-complete yet:

- authenticated SaaS backend
- full database service
- raw annual PDF indexing for all years
- automated PDF evidence snippet extraction from every report
- final hosted LLM inference API
- compliance-certified ESG report generation

## 12. Suggested Founder Office Submission Folder

Founder Office can make a final ZIP with this structure:

```text
CarbonTatvaAI_Final_Submission/
  OVERVIEW.md
  CarbonTatvaAI-ESG-Intelligence-main.zip
  CarbonTatvaAI_ESG_Benchmark_Dashboard_Share.zip
  links.txt
  optional_artifacts/
    sample benchmark reports
    screenshots
    dataset manifest files
```

Suggested `links.txt` content:

```text
GitHub Repo:
https://github.com/propixx/CarbonTatvaAI-ESG-Intelligence

GitHub Repo ZIP:
https://github.com/propixx/CarbonTatvaAI-ESG-Intelligence/archive/refs/heads/main.zip

Deployed Dashboard:
https://propixx.github.io/CarbonTatvaAI-ESG-Intelligence/dashboard/

Literature Review:
https://docs.google.com/spreadsheets/d/1DLYi0VMLkZ1fcprrTtyAzilIK01IgsiuU4mhVYl1zBM/edit?usp=sharing

PRD Dataset Drive:
https://drive.google.com/drive/folders/1HgnCVge9d9-EWDv9QbOiRMrFRylmPge5?usp=drive_link

Additional Report/Data Drive:
https://drive.google.com/drive/folders/18xfBj5-S4erVfAhyFHz7PSSZZtWjRhvg?usp=drive_link
```

## 13. Meeting Update Version

Short update:

> I have pushed the ESG benchmarking engine repo and deployed the static dashboard. It supports target company/year selection, sector-wise comparison, custom peer group comparison, KPI benchmark stats, missing KPI opportunities, disclosure gaps, and report-source coverage.

Slightly longer update:

> The system currently uses PRD/BRSR structured datasets, indexed BRSR reports from 2021-26, and annual-report extracted evidence. The dashboard is static and shareable through GitHub Pages, while the backend scripts prepare benchmark tables, connect report evidence, and export dashboard data.

Technical update:

> I separated deterministic benchmarking from the LLM layer. Median, average, rank, peer range, and gap detection are computed from structured data. LLM/RAG can later be added on top for evidence-backed explanations and report wording.

## 14. Final Internship Story

The internship started with ESG + LLM research and fine-tuning exploration. I studied how ESG report generation, ESG extraction, RAG, and domain-specific models are used in sustainability workflows. I then worked on KPI-to-ESG summary generation using Llama 3.1 8B and Unsloth QLoRA.

As the product direction became clearer, I shifted focus from pure report generation to ESG benchmarking. I organized available PRD/BRSR datasets, connected BRSR and annual-report evidence, built comparison logic for company-vs-sector and company-vs-custom peer groups, and implemented a static dashboard that makes these benchmarks visible.

The final result is a working ESG Intelligence and Benchmarking Engine V1. It helps users compare a company's ESG/BRSR disclosures against peers, identify missing KPIs, detect disclosure gaps, and understand report evidence coverage. This creates a reliable structured base for future LLM/RAG-based recommendations and chatbot functionality.

## 15. Best One-Line Explanation

> I built the benchmark layer underneath an ESG AI product: structured KPI comparison, peer ranking, missing disclosure detection, and evidence coverage, with LLM/RAG positioned as the next explanation layer.

