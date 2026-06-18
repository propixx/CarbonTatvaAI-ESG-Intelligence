# Annual Report Scraping Runbook

## Goal

Collect annual reports for more years for the ESG Intelligence & Benchmarking
Engine, similar to the annual-report/BRSR folders shared in Drive.

The scraper uses NSE's public corporate-filings annual-report endpoint:

`/api/annual-reports?index=equities&symbol=SYMBOL`

## Smoke Test

Run metadata-only first:

```powershell
python scrape_nse_annual_reports.py --symbols RELIANCE,TCS,INFY --years 2025-26,2024-25
```

Then download for those symbols:

```powershell
python scrape_nse_annual_reports.py --symbols RELIANCE,TCS,INFY --years 2025-26,2024-25,2023-24 --download
```

Output:

```text
data/scraped_annual_reports/
```

Important files:

- `nse_annual_reports_manifest.csv`
- `nse_annual_reports_manifest.json`
- `files/<year>/*.pdf` or `*.zip`

## Larger Run

Use symbols from the existing KPI summary company map:

```powershell
python scrape_nse_annual_reports.py --limit 25 --years 2025-26,2024-25 --download
```

Then increase gradually:

```powershell
python scrape_nse_annual_reports.py --limit 50 --years 2025-26,2024-25 --download
```

Full run:

```powershell
python scrape_nse_annual_reports.py --years 2025-26,2024-25,2023-24,2022-23 --download
```

## Resume Behavior

The scraper is resumable:

- existing files larger than 20 KB are skipped,
- manifest is rewritten after each symbol,
- failed symbols are marked with `query_error` or `download_error`.

## What To Say

> I started the annual-report scraping step using NSE's public corporate-filings
> annual-report endpoint. The scraper takes NSE symbols from our existing company
> map, downloads annual reports year-wise, and writes a manifest so we can track
> which company/year reports were collected and resume failed downloads.
