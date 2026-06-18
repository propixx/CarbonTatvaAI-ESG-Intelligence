#!/usr/bin/env python3
"""Scrape NSE annual-report metadata and optionally download report files.

This is the first data-gathering step for the ESG Intelligence & Benchmarking
Engine. It uses NSE's public corporate-filings annual-report endpoint and a
symbol list derived from the existing KPI summary company map.

Example:
    python scrape_nse_annual_reports.py --symbols RELIANCE,TCS,INFY --download
    python scrape_nse_annual_reports.py --limit 50 --years 2025-26,2024-25 --download
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "scraped_annual_reports"
DEFAULT_SYMBOL_MAP = ROOT / "data" / "kpi_summary" / "company_name_map.json"
NSE_HOME = "https://www.nseindia.com"
NSE_ANNUAL_REPORTS_PAGE = (
    "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports"
)
NSE_ANNUAL_REPORTS_API = "https://www.nseindia.com/api/annual-reports"


def clean_symbol(value: Any) -> str:
    text = "" if value is None else str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9&-]+", "", text)
    return text


def year_label(from_year: Any, to_year: Any) -> str:
    from_text = str(from_year).strip()
    to_text = str(to_year).strip()
    if len(from_text) == 4 and len(to_text) == 4:
        return f"{from_text}-{to_text[-2:]}"
    return f"{from_text}-{to_text}"


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def load_symbols(symbol_map: Path, explicit_symbols: str | None, limit: int | None) -> list[str]:
    symbols: set[str] = set()
    if explicit_symbols:
        symbols.update(clean_symbol(item) for item in explicit_symbols.split(","))
        ordered = sorted(symbol for symbol in symbols if symbol)
        return ordered[:limit] if limit else ordered

    if symbol_map.exists():
        data = json.loads(symbol_map.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for item in data.values():
                if isinstance(item, dict):
                    symbol = clean_symbol(item.get("nse_symbol"))
                    if symbol:
                        symbols.add(symbol)
                    company = clean_symbol(item.get("company"))
                    # Many mapped company values are already NSE symbols.
                    if company and len(company) <= 20 and "LIMITED" not in company:
                        symbols.add(company)

    ordered = sorted(symbol for symbol in symbols if symbol)
    if limit:
        ordered = ordered[:limit]
    return ordered


def nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": NSE_ANNUAL_REPORTS_PAGE,
        }
    )
    # NSE usually requires a page visit to set cookies before API calls.
    session.get(NSE_ANNUAL_REPORTS_PAGE, timeout=30)
    return session


def query_annual_reports(session: requests.Session, symbol: str, index: str = "equities") -> list[dict[str, Any]]:
    params = {"index": index, "symbol": symbol}
    response = session.get(NSE_ANNUAL_REPORTS_API, params=params, timeout=30)
    if response.status_code in {401, 403}:
        session.get(NSE_ANNUAL_REPORTS_PAGE, timeout=30)
        response = session.get(NSE_ANNUAL_REPORTS_API, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    return rows if isinstance(rows, list) else []


def download_file(session: requests.Session, url: str, output_path: Path) -> tuple[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 20_000:
        return "already_exists", output_path.stat().st_size

    headers = {
        "User-Agent": session.headers["User-Agent"],
        "Referer": NSE_ANNUAL_REPORTS_PAGE,
        "Accept": "application/pdf,application/zip,application/octet-stream,*/*",
    }
    with session.get(url, headers=headers, stream=True, timeout=90) as response:
        response.raise_for_status()
        temp_path = output_path.with_suffix(output_path.suffix + ".part")
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if chunk:
                    handle.write(chunk)
        size = temp_path.stat().st_size
        if size < 20_000:
            temp_path.unlink(missing_ok=True)
            return "too_small", size
        temp_path.replace(output_path)
        return "downloaded", size


def flatten_row(symbol: str, row: dict[str, Any]) -> dict[str, Any]:
    report_url = str(row.get("fileName") or "")
    label = year_label(row.get("fromYr"), row.get("toYr"))
    return {
        "symbol": symbol,
        "company_name": row.get("companyName"),
        "from_year": row.get("fromYr"),
        "to_year": row.get("toYr"),
        "reporting_year": label,
        "submission_type": row.get("submission_type"),
        "broadcast_datetime": row.get("broadcast_dttm"),
        "dissemination_datetime": row.get("disseminationDateTime"),
        "file_size_text": row.get("attFileSize"),
        "report_url": report_url,
        "source": "NSE annual-reports API",
    }


def output_path_for(base_dir: Path, row: dict[str, Any]) -> Path:
    url = row["report_url"]
    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path).suffix.lower() or ".pdf"
    name = safe_filename(
        f"{row['symbol']}_{row['reporting_year']}_{Path(parsed.path).stem}"
    )
    return base_dir / row["reporting_year"] / f"{name}{suffix}"


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol",
        "company_name",
        "from_year",
        "to_year",
        "reporting_year",
        "submission_type",
        "broadcast_datetime",
        "dissemination_datetime",
        "file_size_text",
        "report_url",
        "local_path",
        "download_status",
        "download_size_bytes",
        "error",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol-map", type=Path, default=DEFAULT_SYMBOL_MAP)
    parser.add_argument("--symbols", help="Comma-separated NSE symbols, e.g. RELIANCE,TCS,INFY")
    parser.add_argument("--years", default="2025-26,2024-25,2023-24,2022-23")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Limit number of symbols for smoke runs")
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--index", default="equities", choices=["equities", "sme"])
    args = parser.parse_args()

    wanted_years = {item.strip() for item in args.years.split(",") if item.strip()}
    symbols = load_symbols(args.symbol_map, args.symbols, args.limit)
    if not symbols:
        raise RuntimeError("No NSE symbols found. Pass --symbols RELIANCE,TCS or provide company_name_map.json.")

    args.output.mkdir(parents=True, exist_ok=True)
    session = nse_session()

    manifest_rows: list[dict[str, Any]] = []
    for idx, symbol in enumerate(symbols, 1):
        print(f"[{idx}/{len(symbols)}] {symbol}")
        try:
            rows = query_annual_reports(session, symbol, index=args.index)
            for raw_row in rows:
                row = flatten_row(symbol, raw_row)
                if wanted_years and row["reporting_year"] not in wanted_years:
                    continue
                row["local_path"] = ""
                row["download_status"] = "metadata_only"
                row["download_size_bytes"] = ""
                row["error"] = ""
                if args.download and row["report_url"]:
                    local_path = output_path_for(args.output / "files", row)
                    try:
                        status, size = download_file(session, row["report_url"], local_path)
                        row["local_path"] = str(local_path.relative_to(args.output))
                        row["download_status"] = status
                        row["download_size_bytes"] = size
                    except Exception as exc:
                        row["download_status"] = "download_error"
                        row["error"] = str(exc)
                manifest_rows.append(row)
        except Exception as exc:
            manifest_rows.append(
                {
                    "symbol": symbol,
                    "company_name": "",
                    "from_year": "",
                    "to_year": "",
                    "reporting_year": "",
                    "submission_type": "",
                    "broadcast_datetime": "",
                    "dissemination_datetime": "",
                    "file_size_text": "",
                    "report_url": "",
                    "local_path": "",
                    "download_status": "query_error",
                    "download_size_bytes": "",
                    "error": str(exc),
                    "source": "NSE annual-reports API",
                }
            )
        write_manifest(args.output / "nse_annual_reports_manifest.csv", manifest_rows)
        time.sleep(args.sleep)

    write_manifest(args.output / "nse_annual_reports_manifest.csv", manifest_rows)
    json_manifest = {
        "symbols_requested": len(symbols),
        "rows_collected": len(manifest_rows),
        "years": sorted(wanted_years),
        "download_enabled": args.download,
        "output": str(args.output),
    }
    (args.output / "nse_annual_reports_manifest.json").write_text(
        json.dumps(json_manifest, indent=2),
        encoding="utf-8",
    )
    print("\nDone.")
    print(json.dumps(json_manifest, indent=2))
    print("Manifest:", args.output / "nse_annual_reports_manifest.csv")


if __name__ == "__main__":
    main()
