# CarbonTatvaAI ESG Benchmarking Engine - Simple Explainer

This document explains what the project is, what has been built, what data is used, how it is related to the PRD, and how to run/share the demo.

## 1. One-Line Explanation

CarbonTatvaAI ESG Benchmarking Engine compares one company's ESG/BRSR KPIs and disclosures against a sector or custom peer group, then shows where the company stands and what gaps it has.

In very simple words:

> Select a company, select a year, select a sector or peer group, and the system tells how that company compares with others on ESG metrics.

## 2. What The PRD Is Asking For

The PRD is not mainly asking for a chatbot first. The main idea is an ESG intelligence and benchmarking product.

The product should help a company answer questions like:

- Are our ESG numbers better or worse than peers?
- Are we missing important BRSR disclosures that peers are reporting?
- What does our sector usually disclose?
- Which peer companies are stronger on specific ESG indicators?
- What gaps should we fix before publishing our ESG/BRSR report?

So the PRD direction is:

```text
Company ESG/BRSR data
    -> compare with sector/custom peers
    -> find KPI gaps and disclosure gaps
    -> show rank/average/median/range
    -> help improve reporting before publication
```

## 3. What We Built

We built a working local dashboard and backend data pipeline for ESG benchmarking.

Current working pieces:

1. Report corpus connector
2. Benchmark company-year database
3. Sector-wise comparison
4. Custom peer-group comparison
5. KPI comparison table
6. Missing KPI opportunities
7. Disclosure gap analysis
8. Report-source coverage view
9. Static shareable dashboard

## 4. What The Dashboard Does

The dashboard has these controls:

### Target Company

This is the company we want to evaluate.

Example:

```text
Aarti Drugs Limited
TCS
Reliance
BEML
```

### Reporting Year

This selects the financial year.

Current years available from the connected data:

- FY 2021-22
- FY 2022-23
- FY 2023-24
- FY 2024-25
- FY 2025-26

### Benchmark Mode

There are two modes.

#### Sector-Wise Comparison

The selected company is compared against companies in the same selected sector.

Example:

```text
Target: Aarti Drugs
Sector: Healthcare
Output: Aarti Drugs vs Healthcare peer group
```

#### Custom Company Group

The user manually selects peer companies.

Example:

```text
Target: TCS
Peers: Infosys, Wipro, HCL, Tech Mahindra
Output: TCS vs selected IT peers
```

This directly answers Mantavya's request:

> "Also include the option to select group of companies / sector."

## 5. Dashboard Tabs

### KPI Comparison

This tab compares numerical ESG KPIs.

For each KPI, it shows:

- Company value
- Peer median
- Peer average
- Peer range
- Company rank
- Interpretation

Example:

```text
Scope 1 emissions:
Company: 10,000 tCO2e
Peer median: 14,000 tCO2e
Rank: 8/30
Interpretation: better than peer median
```

### Missing KPIs

This tab shows KPIs that the selected company has not disclosed but peers have disclosed.

Example:

```text
Renewable energy share:
65% of selected peers disclose this KPI.
Suggestion: Track and disclose renewable energy share.
```

Meaning:

> If many peer companies disclose a KPI and our selected company does not, it is a reporting gap.

### Disclosure Gaps

This tab checks ESG topic coverage.

Examples of disclosure topics:

- Environmental disclosure
- Social disclosure
- Governance disclosure
- Climate risk
- Net-zero target
- Scope 1
- Scope 2
- Scope 3
- Water
- Waste
- Diversity
- Board governance

It finds cases where peers commonly disclose a topic but the target company does not.

### Report Sources

This tab shows whether the selected company-year has connected BRSR/annual evidence.

It shows:

- BRSR reports connected
- Annual report sources connected
- Total connected sources
- Example source path
- Peer evidence coverage

This is important because it proves the dashboard is connected to actual report/evidence inventory, not just a mock UI.

### Visual View

This gives a simple bar-style comparison for selected KPIs.

It is useful for demo/screenshots.

## 6. What Data Was Used

### A. PRD/BRSR Master Dataset

This is the main structured benchmark dataset.

It contains:

- Company name
- Reporting year
- Sector
- ESG/BRSR disclosure flags
- KPI values
- Framework/assurance metadata where available

Used for:

- KPI comparison
- Sector grouping
- Peer grouping
- Disclosure gap analysis
- Benchmark company-year table

### B. BRSR Reports From 2021-24 Zip

File used:

```text
C:\Users\Pratyush\Downloads\brsr 2021-24.zip
```

This zip was indexed directly without extracting all PDFs.

It contains:

```text
FY 2021-22: 177 BRSR PDFs
FY 2022-23: 801 BRSR PDFs
FY 2023-24: 1059 BRSR PDFs
```

Why this matters:

> We do not need to extract or upload this huge zip manually. The connector reads the zip file names and indexes the PDF members.

### C. Scraped BRSR Reports For 2024-25 And 2025-26

These were already present locally under the older SusGen workspace.

Connected counts:

```text
FY 2024-25: 2182 BRSR report entries
FY 2025-26: 192 BRSR report entries
```

### D. Annual Report Extracted Evidence

Raw annual-report PDFs are very large, so we did not download/store all of them.

Instead, the pipeline uses already extracted annual-report evidence:

- annual-report KPI extraction files
- annual-report paragraph/intent files

Connected annual evidence:

```text
1402 annual-report extracted evidence rows
```

This is the practical method for now:

```text
Do not download all annual PDFs
Use extracted annual KPI/paragraph evidence
Keep raw annual PDF fetching as an on-demand future step
```

## 7. Final Corpus Connected

Final connected corpus:

```text
5813 total report/evidence rows
4411 BRSR report entries
1402 annual-report extracted evidence entries
3864 benchmark company-year rows linked to evidence
```

Breakdown:

```text
BRSR FY 2021-22: 177
BRSR FY 2022-23: 801
BRSR FY 2023-24: 1059
BRSR FY 2024-25: 2182
BRSR FY 2025-26: 192
Annual extracted evidence FY 2022-23: 1400
Other annual extracted metadata/evidence: 2
```

## 8. What Each Main Script Does

### `scripts/benchmarking/connect_report_corpus.py`

This script connects available report/evidence files to benchmark company-year rows.

It scans:

```text
..\SusGen\data
~\Downloads\brsr 2021-24.zip
```

It creates:

```text
artifacts/benchmark_engine/report_corpus_index.csv
artifacts/benchmark_engine/benchmark_company_year_report_links.csv
artifacts/benchmark_engine/report_corpus_manifest.json
```

Meaning:

> This script makes the system aware of which BRSR/annual evidence exists for which company/year.

### `scripts/benchmarking/export_dashboard_data.py`

This script converts benchmark CSVs into a JavaScript data file used by the static dashboard.

It creates:

```text
dashboard/dashboard_data.js
```

Meaning:

> The dashboard can run without a Python backend because all required benchmark data is exported into this JS file.

### `scripts/benchmarking/comparative_benchmark.py`

This is the command-line benchmark engine.

It can compare:

- company vs sector
- company vs custom peers

Example:

```powershell
python scripts/benchmarking/comparative_benchmark.py --company "TCS" --year "FY 2024-25"
```

Custom peers:

```powershell
python scripts/benchmarking/comparative_benchmark.py --company "TCS" --year "FY 2024-25" --peers "INFOSYS,WIPRO,HCL,TECH MAHINDRA"
```

### `dashboard/index.html`

The main UI page.

### `dashboard/styles.css`

The dashboard styling.

### `dashboard/app.js`

The frontend logic:

- company dropdown
- year dropdown
- sector/custom peer mode
- KPI comparison
- gaps
- report-source coverage
- chart view

### `dashboard/dashboard_data.js`

The exported benchmark data used by the dashboard.

This is why the dashboard can be shared as a simple static folder.

## 9. How It Is Related To The PRD

The PRD wants an ESG intelligence and benchmarking system.

This implementation covers the V1 of that:

| PRD Need | Current Implementation |
|---|---|
| Benchmark disclosures against peers | Sector/custom peer comparison |
| Identify gaps | Missing KPI and disclosure gap tabs |
| Compare industry leaders | Peer average, median, range, rank |
| Support custom peer group | Custom company group selector |
| Use BRSR/annual data | BRSR zip, scraped BRSR, annual extracted evidence connected |
| Help improve reporting before publication | Gap suggestions and peer adoption indicators |
| Show evidence coverage | Report Sources tab |

So the current dashboard is not just a random UI. It is a V1 implementation of the benchmarking logic described in the PRD.

## 10. What This Is Not Yet

This is important to say honestly.

Current dashboard is:

```text
Static benchmark dashboard + connected report/evidence index
```

It is not yet:

- a full production backend
- a login-based SaaS app
- a full PDF question-answering chatbot
- a complete report-generation model
- a legal/compliance-grade ESG audit tool
- a full raw annual-report parser for all years

Those can be future stages.

## 11. Why There Is No Drag And Drop

Earlier idea:

```text
User uploads or drags report files manually.
```

Current better approach:

```text
The system automatically indexes available local report folders and zips.
```

So the dashboard no longer needs upload/drag-drop.

Why this is better:

- less manual work
- more repeatable
- easier to demo
- avoids browser upload problems
- can index large zips without extracting them

## 12. How To Run Locally

Open PowerShell inside the repo:

```powershell
cd C:\Users\Pratyush\Documents\Codex\2026-05-26\files-mentioned-by-the-user-esg\CarbonTatvaAI-ESG-Intelligence
```

Regenerate corpus links:

```powershell
python scripts/benchmarking/connect_report_corpus.py
```

Regenerate dashboard data:

```powershell
python scripts/benchmarking/export_dashboard_data.py
```

Start local dashboard:

```powershell
python -m http.server 8899 --directory dashboard
```

Open in browser:

```text
http://127.0.0.1:8899
```

## 13. How To Share

There are three practical sharing options.

### Option 1: Send Dashboard Zip

Use the generated ZIP:

```text
CarbonTatvaAI_ESG_Benchmark_Dashboard_Share.zip
```

The receiver can unzip it and open:

```text
index.html
```

This is the fastest sharing method.

### Option 2: GitHub Pages

Because the dashboard is static HTML/CSS/JS, it can be hosted using GitHub Pages.

Expected URL format after enabling GitHub Pages:

```text
https://propixx.github.io/CarbonTatvaAI-ESG-Intelligence/dashboard/
```

Steps:

1. Go to GitHub repo.
2. Settings.
3. Pages.
4. Source: Deploy from branch.
5. Branch: `main`.
6. Folder: `/root`.
7. Save.

Then open:

```text
https://propixx.github.io/CarbonTatvaAI-ESG-Intelligence/dashboard/
```

### Option 3: Deploy On Netlify/Vercel

Upload the `dashboard` folder or connect the GitHub repo.

Build command:

```text
None
```

Publish directory:

```text
dashboard
```

This is also easy because there is no backend required.

## 14. What To Say In A Meeting

Short update:

> I built a real-data ESG benchmarking dashboard. It supports selecting a target company/year and comparing it against either a sector or a custom peer group. It shows KPI value, peer median, peer average, range, rank, missing KPI opportunities, disclosure gaps, and connected BRSR/annual evidence coverage.

More detailed update:

> I connected the benchmark engine with the available PRD/BRSR master data, the BRSR 2021-24 zip, scraped BRSR reports for 2024-25 and 2025-26, and annual-report extracted evidence. The dashboard is static and shareable, but the data is real. It can be run locally or hosted through GitHub Pages/Netlify.

If asked about annual reports:

> Raw annual PDFs are too large to include directly, so for now I used annual-report extracted KPI and paragraph/intent evidence. The connector is written so raw annual PDFs can be indexed later if they are added.

If asked about drag/drop:

> I removed the need for drag/drop in the dashboard. The data is connected through an automatic corpus-indexing script, including large zip files without extraction.

## 15. Current GitHub Repo

```text
https://github.com/propixx/CarbonTatvaAI-ESG-Intelligence
```

## 16. Final Mental Model

Think of the project like this:

```text
Reports + PRD master data
        |
        v
Corpus index + benchmark table
        |
        v
Company vs sector/custom peers
        |
        v
KPI comparison + disclosure gaps + evidence coverage
        |
        v
Better ESG reporting decisions
```

That is the core of the ESG Intelligence & Benchmarking Engine.
