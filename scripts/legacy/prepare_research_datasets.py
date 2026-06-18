#!/usr/bin/env python3
"""Prepare optional public ESG research datasets for CarbonTatvaAI.

This script is intentionally conservative:
- SusGen and the user CSV are safe defaults.
- ESG-CID, DynamicESG, ESG-Activities, and MMESGBench are optional because
  license/use-case constraints differ.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import requests

from convert_esg_csv_to_susgen import make_examples as make_csv_examples

ALLOW_INSECURE_DOWNLOADS = False


RAW_ESG_ACTIVITIES = {
    "train_original": "https://raw.githubusercontent.com/Mattia-Brt/Fine_tuning_LLM/main/data/train_o.json",
    "train_original_synthetic": "https://raw.githubusercontent.com/Mattia-Brt/Fine_tuning_LLM/main/data/train_o%2Bs.json",
    "test": "https://raw.githubusercontent.com/Mattia-Brt/Fine_tuning_LLM/main/data/test.json",
}

RAW_DYNAMIC_ESG = "https://raw.githubusercontent.com/ymntseng/DynamicESG/master/DynamicESG_dataset.json"


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fetch_json(url: str):
    response = requests.get(url, timeout=120, verify=not ALLOW_INSECURE_DOWNLOADS)
    response.raise_for_status()
    return response.json()


def normalize_instruction_items(items):
    output = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if {"instruction", "input", "output"} <= set(item):
            output.append(
                {
                    "instruction": str(item.get("instruction", "")),
                    "input": str(item.get("input", "")),
                    "output": str(item.get("output", "")),
                }
            )
    return output


def load_susgen(max_items: int | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets first: pip install datasets") from exc

    for repo_id in ("WHATX/SusGen-30k", "WHATX/SusGen-30K"):
        try:
            dataset = load_dataset(repo_id)
            split = dataset["train"] if "train" in dataset else dataset[next(iter(dataset.keys()))]
            items = normalize_instruction_items(split)
            return items[:max_items] if max_items else items
        except Exception as exc:
            print(f"Could not load {repo_id}: {exc}")
    raise RuntimeError("Could not download SusGen from HuggingFace.")


def load_local_csv(csv_path: Path, max_items: int | None):
    import csv

    if not csv_path.exists():
        print(f"CSV not found, skipping: {csv_path}")
        return []
    examples = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            examples.extend(make_csv_examples(row, "all"))
            if max_items and len(examples) >= max_items:
                return examples[:max_items]
    return examples


def load_esg_activities(split_name: str, max_items: int | None):
    url = RAW_ESG_ACTIVITIES[split_name]
    data = fetch_json(url)
    examples = normalize_instruction_items(data)
    for item in examples:
        item["instruction"] = (
            "Classify whether the text aligns with the ESG activity described. "
            "Answer only 1 for aligned or 0 for not aligned.\n\n"
            + item["instruction"]
        )
    return examples[:max_items] if max_items else examples


def consensus(value):
    if isinstance(value, list):
        if len(value) == 0:
            return ""
        if len(set(map(str, value))) == 1:
            return str(value[0])
        return " / ".join(map(str, value))
    return "" if value is None else str(value)


def load_dynamic_esg(max_items: int | None):
    data = fetch_json(RAW_DYNAMIC_ESG)
    examples = []
    for item in data:
        headline = item.get("News_Headline", "")
        url = item.get("URL", "")
        text_input = f"News headline: {headline}\nURL: {url}"
        impact_type = consensus(item.get("Impact_Type"))
        impact_duration = consensus(item.get("Impact_Duration"))
        esg_category = consensus(item.get("ESG_Category"))
        if impact_type:
            examples.append(
                {
                    "instruction": "Classify this ESG news item by impact type.",
                    "input": text_input,
                    "output": impact_type,
                }
            )
        if impact_duration:
            examples.append(
                {
                    "instruction": "Classify this ESG news item by expected impact duration.",
                    "input": text_input,
                    "output": impact_duration,
                }
            )
        if esg_category:
            examples.append(
                {
                    "instruction": "Classify this ESG news item into ESG category codes.",
                    "input": text_input,
                    "output": esg_category,
                }
            )
        if max_items and len(examples) >= max_items:
            return examples[:max_items]
    return examples


def first_existing_field(row, names):
    for name in names:
        if name in row:
            return name
    lowered = {key.lower(): key for key in row}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def field_containing(row, *needles):
    for key in row:
        lowered = key.lower()
        if all(needle in lowered for needle in needles):
            return key
    return None


def load_esg_cid(max_items: int | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets first: pip install datasets") from exc

    errors = []
    for repo_id in ("airefinery/esg_cid_retrieval", "esgllm/esg_cid_retrieval"):
        try:
            documents = load_dataset(repo_id, "documents")
            queries = load_dataset(repo_id, "queries")
            triplets = load_dataset(repo_id, "triplets")
            break
        except Exception as exc:
            errors.append(f"{repo_id}: {exc}")
    else:
        raise RuntimeError("Could not load ESG-CID:\n" + "\n".join(errors))

    doc_rows = list(documents["train"])
    query_rows = list(queries["train"])
    triplet_rows = list(triplets["train"])
    if not doc_rows or not query_rows or not triplet_rows:
        return []

    doc_id_field = first_existing_field(doc_rows[0], ["chunk_id", "doc_id", "document_id", "id"])
    doc_text_field = first_existing_field(doc_rows[0], ["chunk_text", "text", "document", "content"])
    query_id_field = first_existing_field(query_rows[0], ["query_id", "id"])
    query_text_field = first_existing_field(query_rows[0], ["query_text", "disclosure_text", "text", "query"])
    triplet_query_field = field_containing(triplet_rows[0], "query") or query_id_field
    positive_field = (
        field_containing(triplet_rows[0], "positive")
        or field_containing(triplet_rows[0], "pos")
        or field_containing(triplet_rows[0], "relevant")
    )
    negative_field = field_containing(triplet_rows[0], "negative") or field_containing(triplet_rows[0], "neg")

    required = [doc_id_field, doc_text_field, query_id_field, query_text_field, triplet_query_field, positive_field]
    if any(field is None for field in required):
        print("Could not infer ESG-CID schema. Documents:", doc_rows[0].keys())
        print("Queries:", query_rows[0].keys())
        print("Triplets:", triplet_rows[0].keys())
        return []

    docs = {str(row[doc_id_field]): str(row[doc_text_field]) for row in doc_rows}
    query_text = {str(row[query_id_field]): str(row[query_text_field]) for row in query_rows}

    examples = []
    for row in triplet_rows:
        query = query_text.get(str(row[triplet_query_field]), "")
        positive_doc = docs.get(str(row[positive_field]), "")
        if query and positive_doc:
            examples.append(
                {
                    "instruction": "Decide whether the report excerpt is relevant evidence for the sustainability disclosure requirement. Answer Relevant or Not relevant.",
                    "input": f"Disclosure requirement:\n{query}\n\nReport excerpt:\n{positive_doc}",
                    "output": "Relevant",
                }
            )
        if negative_field:
            negative_doc = docs.get(str(row[negative_field]), "")
            if query and negative_doc:
                examples.append(
                    {
                        "instruction": "Decide whether the report excerpt is relevant evidence for the sustainability disclosure requirement. Answer Relevant or Not relevant.",
                        "input": f"Disclosure requirement:\n{query}\n\nReport excerpt:\n{negative_doc}",
                        "output": "Not relevant",
                    }
                )
        if max_items and len(examples) >= max_items:
            return examples[:max_items]
    return examples


def clone_mmesgbench(target_dir: Path) -> Path:
    if not target_dir.exists():
        subprocess.check_call(
            ["git", "clone", "--depth", "1", "https://github.com/Zhanglei1103/MMESGBench.git", str(target_dir)]
        )
    return target_dir


def export_mmesgbench_eval(target_dir: Path, output_path: Path):
    repo = clone_mmesgbench(target_dir)
    qa_items = []
    for path in (repo / "dataset").rglob("*.json"):
        try:
            data = load_json(path)
        except Exception:
            continue
        if isinstance(data, dict):
            data = data.get("data") or data.get("questions") or [data]
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict) or "question" not in item or "answer" not in item:
                continue
            qa_items.append(
                {
                    "doc_id": item.get("doc_id", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "evidence_pages": item.get("evidence_pages", ""),
                    "evidence_sources": item.get("evidence_sources", ""),
                    "answer_format": item.get("answer_format", ""),
                }
            )
    save_json(output_path, qa_items)
    return qa_items


def main():
    global ALLOW_INSECURE_DOWNLOADS

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/carbontatva_research_training.json")
    parser.add_argument("--csv", default="data/esg_prd_master_dataset_25-26.csv")
    parser.add_argument("--max-per-source", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--include-susgen", action="store_true")
    parser.add_argument("--include-csv", action="store_true")
    parser.add_argument("--include-esg-cid", action="store_true")
    parser.add_argument("--include-esg-activities", action="store_true")
    parser.add_argument("--include-dynamic-esg", action="store_true")
    parser.add_argument("--export-mmesgbench-eval", action="store_true")
    parser.add_argument(
        "--allow-insecure-downloads",
        action="store_true",
        help="Disable TLS verification for local machines with broken certificate stores. Do not use unless needed.",
    )
    args = parser.parse_args()
    ALLOW_INSECURE_DOWNLOADS = args.allow_insecure_downloads

    all_examples = []
    source_counts = {}

    def add(name, items):
        source_counts[name] = len(items)
        all_examples.extend(items)
        save_json(Path("data") / f"{name}.json", items)
        print(f"{name}: {len(items)} examples")

    if args.include_susgen:
        add("susgen_public", load_susgen(args.max_per_source))
    if args.include_csv:
        add("user_brsr_csv", load_local_csv(Path(args.csv), args.max_per_source))
    if args.include_esg_cid:
        add("esg_cid_relevance", load_esg_cid(args.max_per_source))
    if args.include_esg_activities:
        add("esg_activities", load_esg_activities("train_original_synthetic", args.max_per_source))
    if args.include_dynamic_esg:
        add("dynamic_esg", load_dynamic_esg(args.max_per_source))
    if args.export_mmesgbench_eval:
        items = export_mmesgbench_eval(Path("data/MMESGBench"), Path("data/mmesgbench_eval.json"))
        print(f"mmesgbench_eval: {len(items)} QA items")

    random.seed(args.seed)
    random.shuffle(all_examples)
    output_path = Path(args.output)
    save_json(output_path, all_examples)
    save_json(output_path.with_suffix(".manifest.json"), source_counts)
    print(f"\nSaved {len(all_examples)} examples to {output_path}")
    print(f"Manifest: {output_path.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
