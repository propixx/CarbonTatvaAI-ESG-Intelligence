#!/usr/bin/env python3
"""Merge the CarbonTatvaAI LoRA adapter into the base model for deployment."""

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--lora-path", default="results/carbontatva-llama31-qlora/final_lora_adapter")
    parser.add_argument("--output-dir", default="results/carbontatva-merged")
    args = parser.parse_args()

    print("Loading base model in FP16 for merge.")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("Loading LoRA adapter.")
    model = PeftModel.from_pretrained(model, args.lora_path)
    print("Merging weights.")
    model = model.merge_and_unload()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(output_dir)

    size_gb = sum(os.path.getsize(output_dir / name) for name in os.listdir(output_dir)) / 1024 / 1024 / 1024
    print(f"Merged model saved to {output_dir} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
