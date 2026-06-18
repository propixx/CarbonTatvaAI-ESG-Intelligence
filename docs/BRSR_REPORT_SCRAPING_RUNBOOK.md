# BRSR Report Scraping Runbook

## Goal

Create the same style of dataset as the shared Kaggle BRSR dataset:

```text
kaggle_format/
  2024_2025/
    3M India Limited.pdf
    Alicon Castalloy Limited.pdf
  2025_2026/
    Reliance Industries Limited.pdf
zips/
  brsr_reports_2024_2025.zip
  brsr_reports_2025_2026.zip
```

Source endpoint:

```text
https://www.nseindia.com/api/corporate-bussiness-sustainabilitiy
```

NSE spells the endpoint as `bussiness-sustainabilitiy`; keep that spelling.

## Smoke Test

Metadata only:

```powershell
python scrape_nse_brsr_reports.py --years 2025-26,2024-25 --symbols RELIANCE,TCS,INFY
```

Tiny download test:

```powershell
python scrape_nse_brsr_reports.py --years 2025-26,2024-25 --limit 5 --download
```

## Full Run

Run this for the missing newer years after the shared 2021-22, 2022-23, 2023-24 dataset:

```powershell
python scrape_nse_brsr_reports.py --years 2025-26,2024-25 --download
```

Output:

```text
data/scraped_brsr_reports/
  kaggle_format/
  raw_downloads/
  zips/
  nse_brsr_manifest.csv
  nse_brsr_manifest.json
```

## Expected Current Counts

As of 18 June 2026, NSE returned approximately:

- FY 2024-25: around 1,150+ BRSR PDFs
- FY 2025-26: around 110+ BRSR PDFs so far

FY 2025-26 is still ongoing, so the count will increase as more companies file.

## Resume Behavior

The script skips already downloaded files larger than 20 KB. If the run stops,
run the same command again and it will continue without re-downloading completed
PDFs.

## What To Say

> I created a BRSR scraper using NSE's BRSR corporate-filings endpoint. It exports
> the reports in the same Kaggle-style structure as the existing dataset: one
> folder per financial year, company-name PDFs inside, and a separate zip for each
> year. I tested it on a small run and it produced `2024_2025` and `2025_2026`
> folders plus year-wise zips.
