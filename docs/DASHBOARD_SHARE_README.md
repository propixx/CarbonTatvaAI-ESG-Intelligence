# CarbonTatvaAI ESG Benchmark Dashboard - Shared Demo

This folder contains a static demo of the CarbonTatvaAI ESG Benchmarking Dashboard.

## How To Open

Double-click:

```text
index.html
```

If your browser blocks local files, run this in the folder:

```powershell
python -m http.server 8899
```

Then open:

```text
http://127.0.0.1:8899
```

## What It Shows

- Select a target company and reporting year.
- Compare the company against a sector or custom peer group.
- View KPI comparison: company value, peer median, peer average, range, and rank.
- View missing KPI opportunities.
- View disclosure gaps.
- View connected BRSR/annual report evidence coverage.

## Data Included

The demo includes a pre-exported static data file:

```text
dashboard_data.js
```

That file was generated from the benchmark artifacts in the repo and includes indexed BRSR/annual evidence counts. The large raw PDFs and zips are not included in this shared dashboard package.

## Source Repo

```text
https://github.com/propixx/CarbonTatvaAI-ESG-Intelligence
```
