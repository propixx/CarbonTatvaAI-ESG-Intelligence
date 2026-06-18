#!/usr/bin/env python3
"""Send company metadata and KPI JSON to the local CarbonTatvaAI Ollama model."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "carbontatva"
DEFAULT_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = (
    "You are CarbonTatvaAI, an ESG reporting analyst. Write a concise, factual "
    "ESG narrative summary using only the supplied company metadata and KPI "
    "data. Mention all material KPI values that are provided, preserve units, "
    "and describe year-on-year movement accurately. Do not invent policies, "
    "initiatives, awards, targets, committees, certifications, framework "
    "alignment, or explanations that are absent from the input. The summary is "
    "a drafting aid, not a complete statutory BRSR report."
)


def label(key: str) -> str:
    return key.removeprefix("kpi_").replace("_", " ").strip().title()


def build_prompt(metadata: dict[str, Any], kpis: dict[str, Any]) -> str:
    metadata_lines = [
        f"- {label(key)}: {value}"
        for key, value in metadata.items()
        if value is not None and str(value).strip()
    ]
    kpi_lines = [
        f"- {label(key)}: {value}"
        for key, value in kpis.items()
        if value is not None and str(value).strip()
    ]
    return (
        "Task:\n"
        "Write one professional ESG narrative summary that can be used as the "
        "starting point for a company report. Cover every supplied KPI, compare "
        "current and previous values where available, and do not add unsupported claims.\n\n"
        "Company metadata:\n"
        + "\n".join(metadata_lines)
        + "\n\nKPI data:\n"
        + "\n".join(kpi_lines)
    )


def load_input(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    kpis = payload.get("kpis")
    if not isinstance(metadata, dict) or not isinstance(kpis, dict):
        raise ValueError("Input JSON must contain object fields named 'metadata' and 'kpis'.")
    if not str(metadata.get("company", "")).strip():
        raise ValueError("metadata.company is required.")
    if len(kpis) < 2:
        raise ValueError("At least two KPI fields are required.")
    return metadata, kpis


def generate(model: str, url: str, metadata: dict[str, Any], kpis: dict[str, Any]) -> str:
    request_body = json.dumps(
        {
            "model": model,
            "system": SYSTEM_PROMPT,
            "prompt": build_prompt(metadata, kpis),
            "stream": False,
            "options": {
                "temperature": 0,
                "repeat_penalty": 1.05,
                "num_ctx": 4096,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Install/start Ollama and confirm "
            "`ollama run carbontatva` works first."
        ) from exc
    return str(result.get("response", "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("generated_esg_summary.txt"))
    args = parser.parse_args()

    metadata, kpis = load_input(args.input_json)
    summary = generate(args.model, args.url, metadata, kpis)
    args.output.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
