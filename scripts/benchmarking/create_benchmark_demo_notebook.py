#!/usr/bin/env python3
"""Create a Kaggle-friendly comparative ESG benchmarking demo notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "notebooks" / "CarbonTatvaAI_ESG_Benchmark_Demo.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


def main() -> None:
    cells = [
        markdown(
            """
            # CarbonTatvaAI ESG Benchmarking Demo

            This notebook shows the V1 benchmarking logic:

            - Select one company and year
            - Compare it with either its sector or a custom peer group
            - Show company value vs peer average, median, min/max range, and rank
            - Flag missing KPI opportunities and disclosure gaps

            This is **not** model training. It is a dashboard/backend prototype for ESG benchmarking.
            """
        ),
        markdown(
            """
            ## 1. Setup

            On Kaggle, turn **Internet ON** before running.  
            If the repo is not already attached as a dataset, this cell clones it from GitHub.
            """
        ),
        code(
            """
            from pathlib import Path
            import os
            import subprocess
            import sys

            REPO_URL = "https://github.com/propixx/CarbonTatvaAI-ESG-Intelligence.git"
            REPO_DIR = Path("CarbonTatvaAI-ESG-Intelligence")

            if not Path("scripts/benchmarking/comparative_benchmark.py").exists():
                if not REPO_DIR.exists():
                    subprocess.check_call(["git", "clone", REPO_URL])
                os.chdir(REPO_DIR)

            ROOT = Path.cwd()
            print("Working directory:", ROOT)
            print("Benchmark script exists:", (ROOT / "scripts/benchmarking/comparative_benchmark.py").exists())
            """
        ),
        markdown(
            """
            ## 2. Load Benchmark Data

            The current sample benchmark table contains company-year rows with extracted KPI/disclosure indicators.
            """
        ),
        code(
            """
            import pandas as pd
            from IPython.display import display, Markdown

            data_path = ROOT / "artifacts/benchmark_engine/benchmark_company_year.csv"
            df = pd.read_csv(data_path)

            print("Rows:", len(df))
            print("Columns:", len(df.columns))
            display(df[["company", "reporting_year", "sector", "kpi_available_count", "disclosure_coverage_count"]].head(10))

            sector_summary = (
                df.groupby("sector", dropna=False)
                .agg(companies=("company", "nunique"), rows=("company", "count"))
                .sort_values("rows", ascending=False)
                .head(12)
                .reset_index()
            )
            display(sector_summary)
            """
        ),
        markdown(
            """
            ## 3. Helper Functions

            These functions call the comparative benchmark backend and display dashboard-like tables.
            """
        ),
        code(
            """
            import sys
            sys.path.insert(0, str(ROOT / "scripts" / "benchmarking"))

            from comparative_benchmark import (
                DEFAULT_DATA_DIR,
                build_report,
                clean_text,
                format_value,
                load_company_year,
                pick_target,
                select_peers,
            )

            company_year = load_company_year(ROOT / "artifacts" / "benchmark_engine")

            def run_benchmark(company, year=None, sector=None, peers=None, adoption_threshold=60):
                target = pick_target(company_year, company, year)
                peer_df, peer_label = select_peers(company_year, target, sector, peers, year)
                return build_report(company_year, target, peer_df, peer_label, adoption_threshold)

            def kpi_comparison_table(report):
                rows = []
                for item in report["kpi_comparisons"]:
                    unit = item.get("unit", "")
                    rows.append({
                        "KPI": item["kpi"],
                        "Company value": format_value(item.get("company_value"), unit),
                        "Peer median": format_value(item.get("peer_median"), unit),
                        "Peer average": format_value(item.get("peer_average"), unit),
                        "Peer range": f"{format_value(item.get('peer_min'), unit)} to {format_value(item.get('peer_max'), unit)}",
                        "Rank": (
                            f"{item.get('rank_among_group_including_company')}/"
                            f"{item.get('group_size_including_company')}"
                            if item.get("rank_among_group_including_company") else "-"
                        ),
                        "Interpretation": item.get("interpretation", item.get("status", "-")),
                    })
                return pd.DataFrame(rows)

            def missing_kpi_table(report):
                return pd.DataFrame(report["missing_kpis"])

            def disclosure_gap_table(report):
                rows = []
                for item in report["disclosure_gaps"]:
                    rows.append({
                        "Disclosure": item["disclosure"],
                        "Peer adoption %": item["peer_adoption_percent"],
                        "Peer examples": ", ".join(item["peer_examples"][:3]),
                        "Suggestion": item["suggestion"],
                    })
                return pd.DataFrame(rows)

            def show_report(report):
                target = report["target"]
                display(Markdown(
                    f"### {target['company']} | {target['reporting_year']}\\n"
                    f"- Sector: **{target['sector']}**\\n"
                    f"- Peer group: **{report['peer_group']['type']}**\\n"
                    f"- Peer companies used: **{report['peer_group']['companies']}**"
                ))
                display(Markdown("#### KPI Comparison"))
                display(kpi_comparison_table(report))
                display(Markdown("#### Missing KPI Opportunities"))
                mk = missing_kpi_table(report)
                display(mk if not mk.empty else Markdown("No missing KPI opportunities detected."))
                display(Markdown("#### Disclosure Gaps"))
                dg = disclosure_gap_table(report)
                display(dg if not dg.empty else Markdown("No high-adoption disclosure gaps detected."))
            """
        ),
        markdown(
            """
            ## 4. Sector-Wise Benchmark Example

            Here, the selected company is compared against companies from the same sector.
            """
        ),
        code(
            """
            sector_report = run_benchmark(company="360 ONE WAM", year="FY 2024-25")
            show_report(sector_report)
            """
        ),
        markdown(
            """
            ## 5. Custom Peer Group Benchmark Example

            Here, the user manually selects peer companies. This is useful when a company wants to compare against specific competitors instead of the full sector.
            """
        ),
        code(
            """
            custom_report = run_benchmark(
                company="TCS",
                year="FY 2024-25",
                peers="INFOSYS,WIPRO,HCL,TECH MAHINDRA",
            )
            show_report(custom_report)
            """
        ),
        markdown(
            """
            ## 6. Quick Chart

            This chart compares company KPI values against peer medians for the custom peer-group example.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt

            chart_df = kpi_comparison_table(custom_report).head(8).copy()
            numeric_rows = []
            for item in custom_report["kpi_comparisons"][:8]:
                numeric_rows.append({
                    "KPI": item["kpi"],
                    "Company": item.get("company_value"),
                    "Peer median": item.get("peer_median"),
                })
            chart_numeric = pd.DataFrame(numeric_rows).dropna()
            ax = chart_numeric.set_index("KPI")[["Company", "Peer median"]].plot(kind="bar", figsize=(12, 5))
            ax.set_title("Company vs Peer Median")
            ax.set_ylabel("Value")
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## 7. What This Demonstrates

            This notebook shows the first backend logic for the PRD requirement:

            > sector-wise or custom-group-wise comparative ESG analysis

            The next dashboard step is to convert these into buttons:

            - Compare emissions with peers
            - Compare water KPIs with peers
            - Compare waste KPIs with peers
            - Show missing KPIs
            - Show disclosure gaps
            - Show peer examples
            """
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
