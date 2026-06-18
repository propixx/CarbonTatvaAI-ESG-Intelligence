#!/usr/bin/env python3
"""Scrape NSE BRSR reports and export Kaggle-style year folders/zips.

The target layout matches the shared Kaggle dataset style:

    output/
      kaggle_format/
        2024_2025/
          3M India Limited.pdf
          Alicon Castalloy Limited.pdf
        2025_2026/
          Reliance Industries Limited.pdf
      zips/
        brsr_reports_2024_2025.zip
        brsr_reports_2025_2026.zip

NSE source endpoint:
    /api/corporate-bussiness-sustainabilitiy

Yes, the endpoint spelling is NSE's spelling.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
import urllib.parse
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "scraped_brsr_reports"
NSE_HOME = "https://www.nseindia.com"
NSE_BRSR_PAGE = (
    "https://www.nseindia.com/companies-listing/"
    "corporate-filings-bussiness-sustainabilitiy-reports"
)
NSE_BRSR_API = "https://www.nseindia.com/api/corporate-bussiness-sustainabilitiy"


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
            "Referer": NSE_BRSR_PAGE,
        }
    )
    session.get(NSE_BRSR_PAGE, timeout=30)
    return session


def parse_year(value: str) -> tuple[int, int]:
    text = value.strip().replace("_", "-")
    match = re.fullmatch(r"(\d{4})-(\d{2}|\d{4})", text)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid year '{value}'. Use format like 2024-25 or 2024_2025."
        )
    start = int(match.group(1))
    end_text = match.group(2)
    end = int(end_text) if len(end_text) == 4 else int(str(start)[:2] + end_text)
    return start, end


def year_label(start: int, end: int, sep: str = "-") -> str:
    if sep == "_":
        return f"{start}_{end}"
    return f"{start}-{str(end)[-2:]}"


def filing_window_for_fy(start: int, end: int) -> tuple[str, str]:
    # Most Indian companies file a BRSR for FY N-(N+1) after that FY closes.
    # NSE date filters are submission dates, so FY 2024-25 is usually filed
    # from 01-04-2025 onward.
    from_date = f"01-04-{end}"
    to_year = end + 1
    to_date = f"31-03-{to_year}"
    today = date.today()
    if today.year <= to_year:
        to_date = today.strftime("%d-%m-%Y")
    return from_date, to_date


def query_brsr(
    session: requests.Session,
    from_date: str | None = None,
    to_date: str | None = None,
    symbol: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, str] = {}
    if from_date and to_date:
        params["from_date"] = from_date
        params["to_date"] = to_date
    if symbol:
        params["symbol"] = symbol.strip().upper()

    response = session.get(NSE_BRSR_API, params=params, timeout=45)
    if response.status_code in {401, 403}:
        session.get(NSE_BRSR_PAGE, timeout=30)
        response = session.get(NSE_BRSR_API, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    return rows if isinstance(rows, list) else []


def normalize_company_name(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text.replace("\ufeff", " ")).strip()
    if not text:
        return "Unknown Company"
    if text.upper() == text:
        text = text.title()
        acronym_fixes = {
            "3M": "3M",
            "Au ": "AU ",
            "Bse": "BSE",
            "Cdsl": "CDSL",
            "Hdfc": "HDFC",
            "Icici": "ICICI",
            "Idbi": "IDBI",
            "Idfc": "IDFC",
            "Iifl": "IIFL",
            "Irctc": "IRCTC",
            "Irfc": "IRFC",
            "Itc": "ITC",
            "Jsw": "JSW",
            "Lic": "LIC",
            "Ltd": "Ltd",
            "Mrf": "MRF",
            "Nse": "NSE",
            "Ntpc": "NTPC",
            "Ongc": "ONGC",
            "Pfc": "PFC",
            "Sbi": "SBI",
            "Srf": "SRF",
            "Tcs": "TCS",
            "Tvs": "TVS",
            "Upl": "UPL",
        }
        for old, new in acronym_fixes.items():
            text = text.replace(old, new)
    return text


def safe_company_filename(company_name: str, suffix: str = ".pdf") -> str:
    text = normalize_company_name(company_name)
    text = text.replace("&", " and ")
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    if not text:
        text = "Unknown Company"
    return f"{text}{suffix}"


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    fy_from = int(row.get("fyFrom") or 0)
    fy_to = int(row.get("fyTo") or 0)
    report_url = str(row.get("attachmentFile") or "")
    xbrl_url = str(row.get("xbrlFile") or "")
    return {
        "symbol": row.get("symbol") or "",
        "company_name": normalize_company_name(row.get("companyName")),
        "fy_from": fy_from,
        "fy_to": fy_to,
        "reporting_year": year_label(fy_from, fy_to),
        "kaggle_year_folder": year_label(fy_from, fy_to, sep="_"),
        "submission_date": row.get("submissionDate") or "",
        "revision_date": row.get("revisionDate") or "",
        "pdf_size_text": row.get("attFileSize") or "",
        "pdf_url": report_url,
        "xbrl_url": xbrl_url,
        "xbrl_size_text": row.get("xbrlFileSize") or "",
        "source": "NSE BRSR API",
    }


def download_file(session: requests.Session, url: str, output_path: Path) -> tuple[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 20_000:
        return "already_exists", output_path.stat().st_size

    headers = {
        "User-Agent": session.headers["User-Agent"],
        "Referer": NSE_BRSR_PAGE,
        "Accept": "application/pdf,application/xml,application/octet-stream,*/*",
    }
    with session.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        temp_path = output_path.with_suffix(output_path.suffix + ".part")
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
        size = temp_path.stat().st_size
        if size < 20_000:
            temp_path.unlink(missing_ok=True)
            return "too_small", size
        temp_path.replace(output_path)
        return "downloaded", size


def unique_path(path: Path, symbol: str | None = None) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    if symbol:
        candidate = path.with_name(f"{stem} - {symbol}{suffix}")
        if not candidate.exists():
            return candidate
    counter = 2
    while True:
        candidate = path.with_name(f"{stem} ({counter}){suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol",
        "company_name",
        "fy_from",
        "fy_to",
        "reporting_year",
        "kaggle_year_folder",
        "submission_date",
        "revision_date",
        "pdf_size_text",
        "pdf_url",
        "xbrl_url",
        "xbrl_size_text",
        "local_pdf_path",
        "kaggle_pdf_path",
        "download_status",
        "download_size_bytes",
        "error",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_year_zip(year_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(year_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(year_dir.parent))


def parse_symbol_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    return symbols or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years",
        default="2025-26,2024-25",
        help="Comma-separated FY labels, e.g. 2025-26,2024-25,2023-24",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--symbols",
        help="Optional comma-separated NSE symbols for smoke tests, e.g. RELIANCE,TCS,INFY",
    )
    parser.add_argument("--limit", type=int, help="Limit rows per year for a smoke download")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-zips", action="store_true")
    args = parser.parse_args()

    requested_years = [parse_year(item) for item in args.years.split(",") if item.strip()]
    symbols = parse_symbol_list(args.symbols)
    session = nse_session()

    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / "raw_downloads"
    kaggle_dir = args.output / "kaggle_format"
    zips_dir = args.output / "zips"

    all_manifest_rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for start, end in requested_years:
        folder = year_label(start, end, sep="_")
        from_date, to_date = filing_window_for_fy(start, end)
        print(f"\nFY {year_label(start, end)} -> NSE filing window {from_date} to {to_date}")

        if symbols:
            raw_rows: list[dict[str, Any]] = []
            for index, symbol in enumerate(symbols, 1):
                print(f"  [{index}/{len(symbols)}] Query {symbol}")
                raw_rows.extend(query_brsr(session, from_date, to_date, symbol=symbol))
                time.sleep(args.sleep)
        else:
            raw_rows = query_brsr(session, from_date, to_date)

        rows = [
            flatten_row(row)
            for row in raw_rows
            if int(row.get("fyFrom") or 0) == start and int(row.get("fyTo") or 0) == end
        ]
        rows.sort(key=lambda item: (item["company_name"], item["symbol"], item["submission_date"]))
        if args.limit:
            rows = rows[: args.limit]
        print(f"  Matched FY rows: {len(rows)}")

        for row_index, row in enumerate(rows, 1):
            url = row["pdf_url"]
            row["local_pdf_path"] = ""
            row["kaggle_pdf_path"] = ""
            row["download_status"] = "metadata_only"
            row["download_size_bytes"] = ""
            row["error"] = ""

            if not url:
                row["download_status"] = "missing_pdf_url"
                all_manifest_rows.append(row)
                continue
            if url in seen_urls:
                row["download_status"] = "duplicate_url_skipped"
                all_manifest_rows.append(row)
                continue
            seen_urls.add(url)

            parsed = urllib.parse.urlparse(url)
            suffix = Path(parsed.path).suffix.lower() or ".pdf"
            raw_name = f"{row['symbol']}_{row['reporting_year']}_{Path(parsed.path).name}"
            raw_path = raw_dir / folder / re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", raw_name)
            kaggle_path = kaggle_dir / folder / safe_company_filename(row["company_name"], suffix=suffix)
            kaggle_path = unique_path(kaggle_path, str(row["symbol"] or ""))

            if args.download:
                try:
                    status, size = download_file(session, url, raw_path)
                    kaggle_path.parent.mkdir(parents=True, exist_ok=True)
                    if raw_path.exists():
                        shutil.copy2(raw_path, kaggle_path)
                    row["local_pdf_path"] = str(raw_path.relative_to(args.output))
                    row["kaggle_pdf_path"] = str(kaggle_path.relative_to(args.output))
                    row["download_status"] = status
                    row["download_size_bytes"] = size
                except Exception as exc:
                    row["download_status"] = "download_error"
                    row["error"] = str(exc)
            all_manifest_rows.append(row)

            if row_index % 25 == 0:
                print(f"    Downloaded/processed {row_index}/{len(rows)}")
            time.sleep(args.sleep)

        write_manifest(args.output / "nse_brsr_manifest.csv", all_manifest_rows)

        year_dir = kaggle_dir / folder
        if args.download and not args.no_zips and year_dir.exists():
            zip_path = zips_dir / f"brsr_reports_{folder}.zip"
            make_year_zip(year_dir, zip_path)
            print(f"  Wrote zip: {zip_path}")

    write_manifest(args.output / "nse_brsr_manifest.csv", all_manifest_rows)
    manifest = {
        "source": NSE_BRSR_API,
        "years_requested": [year_label(start, end) for start, end in requested_years],
        "rows_collected": len(all_manifest_rows),
        "download_enabled": args.download,
        "kaggle_format_dir": str(kaggle_dir),
        "zips_dir": str(zips_dir),
    }
    (args.output / "nse_brsr_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print("\nDone.")
    print(json.dumps(manifest, indent=2))
    print("CSV manifest:", args.output / "nse_brsr_manifest.csv")


if __name__ == "__main__":
    main()
