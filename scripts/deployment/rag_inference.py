#!/usr/bin/env python3
"""CarbonTatvaAI RAG over local PDF/TXT company reports using the LoRA model."""

import argparse
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--lora-path", default="results/carbontatva-llama31-qlora/final_lora_adapter")
    parser.add_argument("--reports-dir", default="company_reports")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunk-words", type=int, default=320)
    parser.add_argument("--chunk-overlap", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    return parser.parse_args()


def read_pdf(path: Path) -> list[tuple[str, str]]:
    pages = []
    reader = PdfReader(str(path))
    for page_idx, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((f"{path.name} page {page_idx}", text))
    return pages


def read_documents(reports_dir: Path) -> list[tuple[str, str]]:
    docs = []
    for path in sorted(reports_dir.rglob("*")):
        if path.suffix.lower() == ".pdf":
            docs.extend(read_pdf(path))
        elif path.suffix.lower() in {".txt", ".md"}:
            docs.append((path.name, path.read_text(encoding="utf-8", errors="ignore")))
    return docs


def chunk_text(source: str, text: str, chunk_words: int, overlap: int) -> list[dict[str, str]]:
    words = text.split()
    chunks = []
    step = max(1, chunk_words - overlap)
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + chunk_words]).strip()
        if len(piece) > 80:
            chunks.append({"source": source, "text": piece})
    return chunks


def load_llm(base_model: str, lora_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for 4-bit RAG inference.")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def build_prompt(question: str, retrieved: list[dict[str, str]]) -> str:
    context = "\n\n".join(
        f"[{idx}] Source: {item['source']}\n{item['text']}" for idx, item in enumerate(retrieved, 1)
    )
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are CarbonTatvaAI, an expert ESG and sustainability report analyst. "
        "Answer using only the provided report excerpts. Cite source numbers when possible, "
        "state when evidence is missing, and structure carbon disclosures clearly.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Report excerpts:\n{context}\n\nQuestion:\n{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory not found: {reports_dir}")

    raw_docs = read_documents(reports_dir)
    if not raw_docs:
        raise RuntimeError(f"No PDF/TXT/MD documents found in {reports_dir}")

    chunks = []
    for source, text in raw_docs:
        chunks.extend(chunk_text(source, text, args.chunk_words, args.chunk_overlap))
    if not chunks:
        raise RuntimeError("Documents were loaded, but no usable text chunks were extracted.")
    print(f"Indexed {len(chunks)} chunks from {len(raw_docs)} document pages/files.")

    embedder = SentenceTransformer(args.embed_model)
    embeddings = embedder.encode([item["text"] for item in chunks], normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    model, tokenizer = load_llm(args.base_model, args.lora_path)
    print("\nRAG pipeline ready. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue
        query_embedding = embedder.encode([question], normalize_embeddings=True)
        scores = embeddings @ np.asarray(query_embedding[0], dtype=np.float32)
        top_indices = np.argsort(scores)[-args.top_k :][::-1]
        retrieved = [chunks[int(index)] | {"score": f"{scores[int(index)]:.4f}"} for index in top_indices]
        prompt = build_prompt(question, retrieved)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"\nCarbonTatvaAI:\n{answer.strip()}\n")
        print("Retrieved sources:")
        for idx, item in enumerate(retrieved, 1):
            print(f"  [{idx}] {item['source']} score={item['score']}")
        print()


if __name__ == "__main__":
    main()
