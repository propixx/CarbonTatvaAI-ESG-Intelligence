# =============================================================================
# 🚀 CarbonTatvaAI — FULL PIPELINE FOR KAGGLE
# =============================================================================
# HOW TO USE:
#   1. Go to kaggle.com → New Notebook
#   2. Settings → Accelerator → GPU T4 x2 (or P100)
#   3. Settings → Internet → ON
#   4. Settings → Add Secret → Name: "HF_TOKEN", Value: your HuggingFace token
#   5. Upload your CSV: Click "+ Add Data" → Upload → esg_prd_master_dataset_25-26.csv
#      (It will appear at /kaggle/input/your-dataset-name/)
#   6. Copy-paste this entire file into the notebook
#   7. Run All Cells
#
# ESTIMATED TIME: ~6-8 hours total on T4
# =============================================================================


# %% [markdown]
# # Cell 1: Install Dependencies

# %%
# IMPORTANT: If you get bitsandbytes CUDA errors after this cell,
# click Runtime → Restart Session, then re-run all cells.

import subprocess, sys

# Install bitsandbytes FIRST with force-reinstall (needs CUDA 12.x compatible version)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.44.0", "--force-reinstall"])

packages = [
    "peft",
    "trl",
    "accelerate",
    "datasets",
    "sentencepiece",
    "protobuf",
    "bert-score",
    "rouge-score",
    "nltk",
    "sentence-transformers",
    "pypdf",
]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✅ All packages installed!")


# %% [markdown]
# # Cell 2: Login to HuggingFace + Setup

# %%
import os, warnings
warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Login using Kaggle secret
from kaggle_secrets import UserSecretsClient
try:
    secrets = UserSecretsClient()
    hf_token = secrets.get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    from huggingface_hub import login
    login(token=hf_token)
    print("✅ Logged into HuggingFace!")
except Exception as e:
    print(f"⚠️ Could not auto-login: {e}")
    print("Run manually: huggingface-cli login")

# Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")


# %% [markdown]
# # Cell 3: Download SusGen-30K Dataset

# %%
import json
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("/kaggle/working/carbontatva")
DATA_DIR = OUTPUT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading SusGen-30K from HuggingFace...")
dataset = load_dataset("WHATX/SusGen-30k")
split = dataset["train"]
print(f"Downloaded {len(split)} examples")

# Save as JSON
susgen_data = [
    {"instruction": str(item["instruction"]),
     "input": str(item["input"]),
     "output": str(item["output"])}
    for item in split
]

susgen_path = DATA_DIR / "susgen_30k.json"
with open(susgen_path, "w", encoding="utf-8") as f:
    json.dump(susgen_data, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(susgen_data)} examples to {susgen_path}")
print(f"   Size: {os.path.getsize(susgen_path)/1024/1024:.1f} MB")
print(f"   Sample: {susgen_data[0]['instruction'][:100]}...")


# %% [markdown]
# # Cell 4: Convert Your ESG CSV (if uploaded)
#
# If you uploaded your CSV as a Kaggle dataset, update the path below.
# If you only want SusGen-30K, skip this cell.

# %%
import csv, re

# ====== UPDATE THIS PATH TO YOUR UPLOADED CSV ======
# Check /kaggle/input/ for your dataset folder name
CSV_PATH = None  # Set to None to skip

# Try to auto-find it
for search_dir in Path("/kaggle/input").iterdir():
    for f in search_dir.rglob("*.csv"):
        if "esg" in f.name.lower() or "brsr" in f.name.lower() or "master" in f.name.lower():
            CSV_PATH = f
            break
    if CSV_PATH:
        break

if CSV_PATH and CSV_PATH.exists():
    print(f"Found CSV: {CSV_PATH}")
    
    def clean(v):
        v = "" if v is None else str(v)
        return re.sub(r"\s+", " ", v.replace("\ufeff", "")).strip()

    def has_val(v):
        return clean(v) not in {"", "nan", "NaN", "None", "null"}

    esg_examples = []
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        print(f"CSV columns ({len(columns)}): {columns[:10]}...")
        
        for row in reader:
            company = clean(row.get("company", row.get("Company", "")))
            year = clean(row.get("reporting_year", row.get("Reporting_Year", "")))
            
            # Build a context string from all available columns
            context_parts = []
            for col in columns:
                val = clean(row.get(col, ""))
                if val and val not in {"nan", "NaN", "None"}:
                    context_parts.append(f"{col}: {val}")
            
            context = "\n".join(context_parts[:30])  # Limit context length
            
            if context:
                # Summary example
                esg_examples.append({
                    "instruction": f"Prepare a structured ESG disclosure summary for this company.",
                    "input": context,
                    "output": f"{company} reported for {year}. Based on the available disclosures, " +
                              "the company's ESG reporting covers the identified sustainability areas " +
                              "as detailed in the input data."
                })
                
                # Carbon example (if emission data exists)
                emission_cols = [c for c in columns if "scope" in c.lower() or "emission" in c.lower() or "co2" in c.lower()]
                emission_data = {c: clean(row.get(c, "")) for c in emission_cols if has_val(row.get(c))}
                if emission_data:
                    emission_context = context + "\n\nEmission data:\n" + "\n".join(
                        f"- {k}: {v}" for k, v in emission_data.items()
                    )
                    esg_examples.append({
                        "instruction": "Summarize the company's greenhouse gas emissions and year-on-year changes.",
                        "input": emission_context,
                        "output": f"For {company} in {year}, the available GHG emissions data indicates: " +
                                  " ".join(f"{k.replace('_',' ')}: {v}." for k, v in emission_data.items())
                    })

    esg_path = DATA_DIR / "esg_csv_instruction.json"
    with open(esg_path, "w", encoding="utf-8") as f:
        json.dump(esg_examples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {len(esg_examples)} examples from CSV → {esg_path}")
else:
    print("⏭️ No ESG CSV found. Using SusGen-30K only.")
    esg_examples = []


# %% [markdown]
# # Cell 5: Merge Datasets into Final Training Data

# %%
import random

all_data = susgen_data.copy()
if esg_examples:
    all_data.extend(esg_examples)
    print(f"Merged: {len(susgen_data)} (SusGen) + {len(esg_examples)} (CSV) = {len(all_data)} total")
else:
    print(f"Using SusGen-30K only: {len(all_data)} examples")

random.seed(2024)
random.shuffle(all_data)

training_path = DATA_DIR / "carbontatva_training.json"
with open(training_path, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"✅ Final training data: {len(all_data)} examples → {training_path}")


# %% [markdown]
# # Cell 6: Fine-Tune Llama 3.1 8B with QLoRA
#
# ⏰ This takes ~6-8 hours on T4. Go sleep.

# %%
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, DataCollatorForLanguageModeling, Trainer
)

# ============================================================
# CONFIG — Tuned for Kaggle T4 (16GB VRAM)
# ============================================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TRAINING_DATA = str(training_path)
RESULTS_DIR = str(OUTPUT_DIR / "results")
NUM_EPOCHS = 3
BATCH_SIZE = 1          # T4 has 16GB — keep this at 1
GRAD_ACCUM = 16         # Effective batch = 1 * 16 = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 384    # Reduced from 512 for T4 memory
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ============================================================
# Load + format data
# ============================================================
print("Loading training data...")
with open(TRAINING_DATA, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

def format_prompt(record):
    system_msg = (
        "You are CarbonTatvaAI, an expert ESG and sustainability report analyst. "
        "Provide detailed, accurate, and well-structured responses about carbon, "
        "climate risk, and ESG disclosures."
    )
    instruction = str(record.get("instruction", "")).strip()
    user_input = str(record.get("input", "")).strip()
    output = str(record.get("output", "")).strip()
    
    if user_input:
        user_msg = f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}"
    else:
        user_msg = instruction
    
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{output}<|eot_id|>"
    )

formatted = [{"text": format_prompt(r)} for r in raw_data]
dataset = Dataset.from_list(formatted)
print(f"✅ {len(dataset)} examples formatted")

# ============================================================
# Load model (4-bit quantized)
# ============================================================
print(f"Loading {MODEL_NAME} with 4-bit quantization...")

use_bf16 = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
print(f"Using {'BF16' if use_bf16 else 'FP16'}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=compute_dtype,
    trust_remote_code=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Model loaded!")

# ============================================================
# Configure LoRA
# ============================================================
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# Tokenize
# ============================================================
def tokenize_batch(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        add_special_tokens=False,
    )
    tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
    return tokens

tokenized_dataset = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
print(f"✅ Tokenized {len(tokenized_dataset)} examples")

# ============================================================
# Train!
# ============================================================
training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    fp16=not use_bf16,
    bf16=use_bf16,
    logging_steps=25,
    save_steps=1000,
    save_total_limit=2,
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    group_by_length=True,
    report_to="none",
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"   Total steps: ~{len(tokenized_dataset) * NUM_EPOCHS // (BATCH_SIZE * GRAD_ACCUM)}")
print(f"   Max seq length: {MAX_SEQ_LENGTH}")
print("=" * 60 + "\n")

trainer.train()

# Save final adapter
LORA_OUTPUT = str(OUTPUT_DIR / "final_lora_adapter")
trainer.model.save_pretrained(LORA_OUTPUT)
tokenizer.save_pretrained(LORA_OUTPUT)
print(f"\n✅ TRAINING COMPLETE! LoRA adapter saved to {LORA_OUTPUT}")

# Free memory
del trainer, model
torch.cuda.empty_cache()
import gc; gc.collect()


# %% [markdown]
# # Cell 7: Test the Trained Model

# %%
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LORA_OUTPUT = str(OUTPUT_DIR / "final_lora_adapter")

print("Loading model for inference...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
)
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base, LORA_OUTPUT)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

test_cases = [
    ("Summarize the greenhouse gas emissions disclosed in the following text.",
     "Tata Steel reported total GHG emissions of 30.5 million tonnes CO2e for FY2024. "
     "Scope 1 was 28.2 MT from blast furnaces. Scope 2 was 2.1 MT from electricity. "
     "Emission intensity was 2.07 tCO2e per tonne of crude steel, down 3.2% YoY."),
    ("Identify the key climate-related risks from this disclosure.",
     "The company's coastal plant faces flood risk from rising sea levels. "
     "EU carbon pricing could raise costs by 15-20%. Water scarcity has intensified."),
    ("Draft a TCFD-aligned governance disclosure.",
     "The Board oversees climate risks via its Sustainability Committee (quarterly). "
     "CEO owns climate strategy. Net-zero by 2050, interim 30% reduction by 2030 from 2020 baseline."),
]

for i, (instruction, text_input) in enumerate(test_cases, 1):
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are CarbonTatvaAI, an expert ESG and sustainability report analyst.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"### Instruction:\n{instruction}\n\n### Input:\n{text_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=512, temperature=0.7, top_p=0.9,
            repetition_penalty=1.15, do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n{'='*60}\nTEST {i}: {instruction[:50]}...\n{'='*60}\n{response.strip()}\n")

print("✅ INFERENCE TEST COMPLETE!")


# %% [markdown]
# # Cell 8: Evaluate (BERTScore, ROUGE-L, BLEU)

# %%
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

NUM_EVAL = 30  # Keep small to finish within session time

with open(str(training_path), "r", encoding="utf-8") as f:
    eval_pool = json.load(f)

random.seed(2024)
eval_samples = random.sample(eval_pool, min(NUM_EVAL, len(eval_pool)))

predictions, references = [], []
for idx, sample in enumerate(eval_samples, 1):
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are CarbonTatvaAI, an expert ESG analyst.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    predictions.append(pred)
    references.append(sample["output"])
    if idx % 10 == 0:
        print(f"  Generated {idx}/{len(eval_samples)}")

print("\nComputing metrics...")
P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
bert_f1 = F1.mean().item()

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l = sum(scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(references, predictions)) / len(predictions)

smooth = SmoothingFunction().method1
bleu = sum(sentence_bleu([r.split()], p.split(), smoothing_function=smooth) for r, p in zip(references, predictions)) / len(predictions)

print(f"\n{'='*40}")
print(f"📊 EVALUATION RESULTS ({NUM_EVAL} samples)")
print(f"{'='*40}")
print(f"  BERTScore F1: {bert_f1:.4f}")
print(f"  ROUGE-L:      {rouge_l:.4f}")
print(f"  BLEU:         {bleu:.4f}")
print(f"{'='*40}")


# %% [markdown]
# # Cell 9: Save Everything to Kaggle Output

# %%
import shutil

# Copy LoRA adapter to /kaggle/working/ (auto-downloadable)
kaggle_output = Path("/kaggle/working/carbontatva-lora-adapter")
if Path(LORA_OUTPUT).exists():
    shutil.copytree(LORA_OUTPUT, kaggle_output, dirs_exist_ok=True)
    print(f"✅ LoRA adapter copied to {kaggle_output}")
    print("   Download from: Kaggle → Output tab → Download All")

# Save eval results
results = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "training_data": f"{len(all_data)} examples",
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE * GRAD_ACCUM,
    "learning_rate": LEARNING_RATE,
    "lora_r": LORA_R,
    "max_seq_length": MAX_SEQ_LENGTH,
    "bert_score_f1": round(bert_f1, 4),
    "rouge_l": round(rouge_l, 4),
    "bleu": round(bleu, 4),
}
with open("/kaggle/working/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ ALL DONE! Results saved to /kaggle/working/eval_results.json")
print(json.dumps(results, indent=2))


# %% [markdown]
# # Cell 10 (OPTIONAL): Push LoRA to HuggingFace Hub
#
# Uncomment and edit the repo_id below to upload your trained adapter.

# %%
# from huggingface_hub import HfApi
# api = HfApi()
# api.upload_folder(
#     folder_path=str(kaggle_output),
#     repo_id="YOUR_USERNAME/CarbonTatvaAI-LoRA",  # ← Change this
#     repo_type="model",
#     create_repo=True,
# )
# print("✅ Uploaded to HuggingFace!")
