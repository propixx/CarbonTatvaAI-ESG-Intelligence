#!/usr/bin/env python3
"""Build a report-corpus index and link reports to benchmark company-years.

The dashboard and benchmark engine should not depend on manual drag/drop.
This script scans known local folders for BRSR PDFs, annual-report PDFs, and
annual-report-derived extracted tables, then creates small index artifacts.
Large PDFs stay outside git; only the lightweight CSV/JSON indexes are written.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "benchmark_engine"
DEFAULT_BENCHMARK = DEFAULT_OUTPUT / "benchmark_company_year.csv"

DEFAULT_ROOTS = [
    ROOT / "data",
    ROOT.parent / "SusGen" / "data",
]

REPORT_SUFFIXES = {".pdf"}
TABLE_SUFFIXES = {".csv", ".json", ".jsonl", ".xlsx", ".xls", ".parquet"}
MAX_COMPANY_TABLE_PARSE_BYTES = 100 * 1024 * 1024


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", " ")
    return re.sub(r"\s+", " ", text).strip()


def safe_relpath(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        pass
    try:
        return str(Path("..") / resolved.relative_to(ROOT.parent.resolve()))
    except ValueError:
        return str(path)


def normalize_company(value: Any) -> str:
    text = clean_text(value).upper()
    text = re.sub(r"\.(PDF|CSV|JSONL?|XLSX?|PARQUET)$", "", text)
    text = re.sub(r"[_-]\d{8,}.*$", "", text)
    text = re.sub(r"\b(BRSR|BUSINESS RESPONSIBILITY|SUSTAINABILITY REPORT|ANNUAL REPORT|FINAL|SIGNED|PDF|REPORT)\b", " ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def loose_company_key(value: Any) -> str:
    text = normalize_company(value)
    text = re.sub(r"\b(LIMITED|LTD|PRIVATE|PVT|INDIA|CO|COMPANY)\b", " ", text)
    text = re.sub(r"[^A-Z0-9]+", "", text)
    return text.strip()


def display_company_from_name(value: str) -> str:
    text = Path(value).stem
    text = re.sub(r"[_-]\d{8,}.*$", "", text)
    text = re.sub(r"\b(BRSR|BUSINESS RESPONSIBILITY|SUSTAINABILITY REPORT|ANNUAL REPORT|FINAL|SIGNED|PDF|REPORT)\b", " ", text, flags=re.I)
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -_")
    return text or Path(value).stem


def infer_year(value: Any) -> str:
    text = clean_text(value)
    patterns = [
        (r"20(2[0-9])[_-]20(2[0-9])", lambda m: f"FY 20{m.group(1)}-{m.group(2)}"),
        (r"20(2[0-9])[_-](2[0-9])", lambda m: f"FY 20{m.group(1)}-{m.group(2)}"),
        (r"(2[0-9])[_-](2[0-9])", lambda m: f"FY 20{m.group(1)}-{m.group(2)}"),
        (r"FY\s*20(2[0-9])\s*[-_/]\s*(2[0-9])", lambda m: f"FY 20{m.group(1)}-{m.group(2)}"),
    ]
    for pattern, formatter in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return formatter(match)
    return ""


def infer_source_type(path_text: str) -> str:
    text = path_text.lower()
    if "brsr" in text or "business responsibility" in text:
        return "brsr_report"
    if "annual" in text:
        if any(marker in text for marker in ["esg_kpis", "paragraph", "intent", ".csv", ".json", ".jsonl", ".xlsx"]):
            return "annual_report_extracted_data"
        return "annual_report"
    return "unknown_report_source"


def iter_report_files(root: Path) -> Iterable[dict[str, Any]]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.lower().endswith(".part"):
            continue
        suffix = path.suffix.lower()
        if suffix in REPORT_SUFFIXES:
            yield {
                "location_type": "file",
                "file_name": path.name,
                "report_path": safe_relpath(path),
                "archive_path": "",
                "archive_member": "",
                "size_bytes": path.stat().st_size,
                "source_text": str(path),
            }


def iter_zip_members(root: Path) -> Iterable[dict[str, Any]]:
    if not root.exists():
        return
    for archive_path in root.rglob("*.zip"):
        if archive_path.name.lower().endswith(".part"):
            continue
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    member_name = Path(info.filename).name
                    if Path(member_name).suffix.lower() not in REPORT_SUFFIXES:
                        continue
                    yield {
                        "location_type": "zip_member",
                        "file_name": member_name,
                        "report_path": "",
                        "archive_path": safe_relpath(archive_path),
                        "archive_member": info.filename,
                        "size_bytes": info.file_size,
                        "source_text": f"{archive_path}/{info.filename}",
                    }
        except zipfile.BadZipFile:
            continue


def iter_annual_extracted_sources(root: Path) -> Iterable[dict[str, Any]]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.lower().endswith(".part") or path.suffix.lower() not in TABLE_SUFFIXES:
            continue
        text = str(path).lower()
        if "annual" not in text:
            continue
        if not any(marker in text for marker in ["esg_kpis", "paragraph", "intent", "metadata", "classification"]):
            continue
        yielded_company_rows = False
        if path.suffix.lower() in {".csv", ".jsonl"} and path.stat().st_size <= MAX_COMPANY_TABLE_PARSE_BYTES:
            try:
                chunks = (
                    pd.read_csv(
                        path,
                        chunksize=50_000,
                        usecols=lambda column: column in {"company", "reporting_year"},
                    )
                    if path.suffix.lower() == ".csv"
                    else [pd.read_json(path, lines=True)[["company", "reporting_year"]]]
                )
                seen_pairs: set[tuple[str, str]] = set()
                for chunk in chunks:
                    if "company" not in chunk:
                        continue
                    for row in chunk.itertuples(index=False):
                        company = clean_text(getattr(row, "company", ""))
                        if not company:
                            continue
                        year = infer_year(getattr(row, "reporting_year", "")) or infer_year(str(path))
                        key = (normalize_company(company), year, path.name)
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        yielded_company_rows = True
                        yield {
                            "location_type": "extracted_table",
                            "file_name": path.name,
                            "report_path": safe_relpath(path),
                            "archive_path": "",
                            "archive_member": "",
                            "size_bytes": path.stat().st_size,
                            "source_text": str(path),
                            "company_display": company,
                            "company_normalized": normalize_company(company),
                            "reporting_year": year,
                        }
            except Exception:
                yielded_company_rows = False

        if not yielded_company_rows:
            yield {
                "location_type": "extracted_table",
                "file_name": path.name,
                "report_path": safe_relpath(path),
                "archive_path": "",
                "archive_member": "",
                "size_bytes": path.stat().st_size,
                "source_text": str(path),
            }


def build_report_index(roots: list[Path], include_zip_members: bool) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        iterators = [iter_report_files(root), iter_annual_extracted_sources(root)]
        if include_zip_members:
            iterators.append(iter_zip_members(root))
        for iterator in iterators:
            for item in iterator:
                source_type = infer_source_type(item["source_text"])
                year = item.get("reporting_year") or infer_year(item["source_text"])
                company_display = item.get("company_display") or display_company_from_name(item["file_name"])
                company_normalized = item.get("company_normalized") or normalize_company(company_display)
                records.append(
                    {
                        "source_type": source_type,
                        "reporting_year": year,
                        "company_display": company_display,
                        "company_normalized": company_normalized,
                        "company_match_key": loose_company_key(company_normalized),
                        "location_type": item["location_type"],
                        "file_name": item["file_name"],
                        "report_path": item["report_path"],
                        "archive_path": item["archive_path"],
                        "archive_member": item["archive_member"],
                        "size_bytes": item["size_bytes"],
                        "source_root": safe_relpath(root),
                    }
                )

    if not records:
        return pd.DataFrame(
            columns=[
                "source_type",
                "reporting_year",
                "company_display",
                "company_normalized",
                "company_match_key",
                "location_type",
                "file_name",
                "report_path",
                "archive_path",
                "archive_member",
                "size_bytes",
                "source_root",
            ]
        )

    df = pd.DataFrame(records)
    df["dedupe_key"] = (
        df["source_type"].astype(str)
        + "|"
        + df["reporting_year"].astype(str)
        + "|"
        + df["company_match_key"].astype(str)
        + "|"
        + df["file_name"].map(lambda value: normalize_company(Path(str(value)).stem))
    )
    location_rank = {"file": 0, "extracted_table": 1, "zip_member": 2}
    df["_location_rank"] = df["location_type"].map(location_rank).fillna(9)
    df = df.sort_values(["dedupe_key", "_location_rank", "size_bytes"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["dedupe_key"], keep="first")
    return df.drop(columns=["dedupe_key", "_location_rank"]).reset_index(drop=True)


def link_to_benchmark(report_index: pd.DataFrame, benchmark_path: Path) -> pd.DataFrame:
    if not benchmark_path.exists() or report_index.empty:
        return pd.DataFrame()
    benchmark = pd.read_csv(benchmark_path)
    if "company_normalized" not in benchmark or "reporting_year" not in benchmark:
        return pd.DataFrame()

    report_groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "brsr_report_count": 0,
            "annual_report_count": 0,
            "annual_extracted_source_count": 0,
            "report_count": 0,
            "report_types_available": set(),
            "example_report_path": "",
        }
    )
    for row in report_index.itertuples(index=False):
        key = (getattr(row, "company_match_key"), getattr(row, "reporting_year"))
        if not key[0] or not key[1]:
            continue
        group = report_groups[key]
        source_type = getattr(row, "source_type")
        group["report_count"] += 1
        group["report_types_available"].add(source_type)
        if source_type == "brsr_report":
            group["brsr_report_count"] += 1
        elif source_type == "annual_report":
            group["annual_report_count"] += 1
        elif source_type == "annual_report_extracted_data":
            group["annual_extracted_source_count"] += 1
        if not group["example_report_path"]:
            group["example_report_path"] = getattr(row, "report_path") or getattr(row, "archive_path")

    linked_rows = []
    for row in benchmark.itertuples(index=False):
        key = (loose_company_key(getattr(row, "company_normalized")), getattr(row, "reporting_year"))
        group = report_groups.get(key)
        if not group:
            continue
        linked_rows.append(
            {
                "company": getattr(row, "company", ""),
                "company_normalized": key[0],
                "reporting_year": key[1],
                "sector": getattr(row, "sector", ""),
                "source_type": getattr(row, "source_type", ""),
                "benchmark_quality_score": getattr(row, "benchmark_quality_score", ""),
                "brsr_report_count": group["brsr_report_count"],
                "annual_report_count": group["annual_report_count"],
                "annual_extracted_source_count": group["annual_extracted_source_count"],
                "report_count": group["report_count"],
                "report_types_available": ", ".join(sorted(group["report_types_available"])),
                "example_report_path": group["example_report_path"],
            }
        )
    return pd.DataFrame(linked_rows)


def write_manifest(output_dir: Path, report_index: pd.DataFrame, linked: pd.DataFrame, roots: list[Path]) -> None:
    by_type_year = []
    if not report_index.empty:
        grouped = report_index.groupby(["source_type", "reporting_year"], dropna=False)
        by_type_year = [
            {
                "source_type": source_type,
                "reporting_year": year,
                "count": int(len(group)),
                "size_mb": round(float(group["size_bytes"].fillna(0).sum()) / 1024 / 1024, 2),
            }
            for (source_type, year), group in grouped
        ]

    manifest = {
        "created_assets": [
            "report_corpus_index.csv",
            "benchmark_company_year_report_links.csv",
            "report_corpus_manifest.json",
        ],
        "roots_scanned": [str(path) for path in roots],
        "report_index_rows": int(len(report_index)),
        "linked_company_year_rows": int(len(linked)),
        "source_type_counts": dict(Counter(report_index["source_type"])) if not report_index.empty else {},
        "counts_by_type_year": by_type_year,
        "notes": [
            "This connects local report files and extracted annual-report evidence to benchmark company-year rows.",
            "Large PDFs/zips are not copied into git; only paths and counts are indexed.",
            "If raw annual-report PDFs are added later, rerun this script and they will be picked up automatically.",
        ],
    }
    (output_dir / "report_corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--include-zip-members",
        action="store_true",
        help="Also inspect PDF members inside zip archives without extracting them. Slower on multi-GB zips.",
    )
    args = parser.parse_args()

    roots = [path for path in args.roots if path.exists()]
    args.output.mkdir(parents=True, exist_ok=True)

    report_index = build_report_index(roots, include_zip_members=args.include_zip_members)
    report_index.to_csv(args.output / "report_corpus_index.csv", index=False)

    linked = link_to_benchmark(report_index, args.benchmark)
    linked.to_csv(args.output / "benchmark_company_year_report_links.csv", index=False)

    write_manifest(args.output, report_index, linked, roots)

    print(f"Report corpus index written to: {args.output / 'report_corpus_index.csv'}")
    print(f"Report/evidence rows indexed: {len(report_index)}")
    print(f"Benchmark company-year rows linked: {len(linked)}")
    if not report_index.empty:
        print("Counts by source type:")
        for source_type, count in Counter(report_index["source_type"]).most_common():
            print(f"- {source_type}: {count}")


if __name__ == "__main__":
    main()
