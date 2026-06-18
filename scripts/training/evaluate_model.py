#!/usr/bin/env python3
"""Evaluate CarbonTatvaAI with BERTScore, ROUGE-L, and BLEU."""

import argparse
import json
import random
from pathlib import Path

import torch
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--lora-path", default="results/carbontatva-llama31-qlora/final_lora_adapter")
    parser.add_argument("--data-path", default="data/carbontatva_training.json")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def prompt_for(sample: dict[str, str]) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are CarbonTatvaAI, an expert ESG and sustainability report analyst.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for 4-bit evaluation.")

    with Path(args.data_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    random.seed(args.seed)
    eval_data = random.sample(data, min(args.num_samples, len(data)))
    print(f"Evaluating on {len(eval_data)} samples.")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    predictions = []
    references = []
    for idx, sample in enumerate(eval_data, 1):
        inputs = tokenizer(
            prompt_for(sample),
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
            add_special_tokens=False,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        predictions.append(pred.strip())
        references.append(sample["output"])
        if idx % 10 == 0:
            print(f"Generated {idx}/{len(eval_data)}")

    print("Computing BERTScore.")
    _, _, f1 = bert_score(predictions, references, lang="en", verbose=True)
    bert_f1 = f1.mean().item()

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(references, predictions)]
    rouge_l = sum(rouge_scores) / len(rouge_scores)

    smooth = SmoothingFunction().method1
    bleu_scores = [
        sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
        for ref, pred in zip(references, predictions)
    ]
    bleu = sum(bleu_scores) / len(bleu_scores)

    print("\nEVALUATION COMPLETE")
    print(f"BERTScore F1: {bert_f1:.4f}")
    print(f"ROUGE-L:      {rouge_l:.4f}")
    print(f"BLEU:         {bleu:.4f}")


if __name__ == "__main__":
    main()
