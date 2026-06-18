#!/usr/bin/env python3
"""Fine-tune Llama 3.1 8B Instruct on CarbonTatva/SusGen data with QLoRA."""

import argparse
import json
import os
import warnings
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data-path", default="data/carbontatva_training.json")
    parser.add_argument("--output-dir", default="results/carbontatva-llama31-qlora")
    parser.add_argument("--num-epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--val-split-ratio", type=float, default=0.005)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--report-to", default=os.environ.get("REPORT_TO", "none"))
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    return parser.parse_args()


def format_prompt(record: dict[str, str]) -> str:
    system_msg = (
        "You are CarbonTatvaAI, an expert ESG and sustainability report analyst. "
        "Provide detailed, accurate, and well-structured responses about carbon, "
        "climate risk, and ESG disclosures."
    )
    instruction = str(record.get("instruction", "")).strip()
    user_input = str(record.get("input", "")).strip()
    output = str(record.get("output", "")).strip()
    user_msg = instruction if not user_input else f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}"
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{output}<|eot_id|>"
    )


def load_training_dataset(data_path: str) -> Dataset:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)
    required = {"instruction", "input", "output"}
    for idx, item in enumerate(raw_data):
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Training item {idx} missing keys: {sorted(missing)}")
    formatted = [{"text": format_prompt(item)} for item in raw_data]
    print(f"Loaded {len(formatted)} training examples from {path}")
    print("First formatted example preview:")
    print(formatted[0]["text"][:500] + "...")
    return Dataset.from_list(formatted)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for QLoRA training. Run this script on a Linux NVIDIA GPU machine.")

    use_bf16 = not args.no_bf16
    if use_bf16 and not torch.cuda.is_bf16_supported():
        print("GPU does not report BF16 support; falling back to FP16.")
        use_bf16 = False
        args.fp16 = True

    dataset = load_training_dataset(args.data_path)
    split = dataset.train_test_split(test_size=args.val_split_ratio, seed=args.seed) if args.val_split_ratio else None
    train_dataset = split["train"] if split else dataset
    eval_dataset = split["test"] if split else None

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print(f"Loading model with 4-bit quantization: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_batch(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            add_special_tokens=False,
        )
        tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
        return tokens

    remove_columns = train_dataset.column_names
    tokenized_train = train_dataset.map(tokenize_batch, batched=True, remove_columns=remove_columns)
    tokenized_eval = eval_dataset.map(tokenize_batch, batched=True, remove_columns=remove_columns) if eval_dataset else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        fp16=args.fp16,
        bf16=use_bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        group_by_length=True,
        report_to=args.report_to,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Starting training.")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_output_dir = Path(args.output_dir) / "final_lora_adapter"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Training complete. LoRA adapter saved to: {final_output_dir}")


if __name__ == "__main__":
    main()
