# =============================================================================
# CarbonTatvaAI - Unsloth PRD Master Fine-Tuning Pipeline
# =============================================================================
# Core task:
#   input  = KPI + metadata + ESG classification + intent labels
#   output = cleaned ESG section/report text
#
# Dataset source:
#   New Drive folder / Kaggle input PRD master dataset.
#
# Default training method:
#   QLoRA with Unsloth, because it is memory efficient.
# =============================================================================


# %% [markdown]
# # Cell 1: Install dependencies

# %%
import subprocess
import sys


def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


pip_install("unsloth")
pip_install("--upgrade", "datasets", "trl", "peft", "accelerate", "huggingface_hub", "sentencepiece", "protobuf")
pip_install("gdown", "pandas", "openpyxl")

print("Dependencies installed.")


# %% [markdown]
# # Cell 2: Login, paths, and config

# %%
import gc
import inspect
import json
import os
import random
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import torch

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OUTPUT_DIR = Path("/kaggle/working/carbontatva_prd_unsloth")
DATA_DIR = OUTPUT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/18xfBj5-S4erVfAhyFHz7PSSZZtWjRhvg?usp=drive_link"

# Options:
#   "qlora" = 4-bit base model + LoRA adapters. More memory efficient.
#   "lora"  = 16-bit base model + LoRA adapters. More VRAM, sometimes faster.
TRAINING_METHOD = "qlora"

MODEL_NAME_16BIT = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME_4BIT = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
USE_16BIT_LORA = TRAINING_METHOD.lower() == "lora"
MODEL_NAME = MODEL_NAME_16BIT if USE_16BIT_LORA else MODEL_NAME_4BIT

MAX_SEQ_LENGTH = 1024
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2 if USE_16BIT_LORA else 1
GRAD_ACCUM = 8 if USE_16BIT_LORA else 16
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
SAVE_STEPS = 50
SAVE_EVERY_MINUTES = 15

# Set 100 or 300 for a smoke test. Keep None for full PRD master training.
MAX_TRAIN_EXAMPLES = None

try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login

    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    login(token=hf_token)
    print("Logged into HuggingFace.")
except Exception as exc:
    print(f"HF_TOKEN Kaggle secret not found or login failed: {exc}")
    print("If running locally, login with: huggingface-cli login")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required. Enable GPU accelerator first.")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Training mode: {'16-bit LoRA' if USE_16BIT_LORA else '4-bit QLoRA'}")
print(f"Model: {MODEL_NAME}")


# %% [markdown]
# # Cell 3: Find/download PRD master dataset files

# %%
import gdown


TABLE_EXTENSIONS = {".csv", ".json", ".jsonl", ".xlsx", ".xls", ".parquet"}


def table_files(root: Path):
    if not root.exists():
        return []
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".part" in path.name.lower():
            continue
        if path.suffix.lower() in TABLE_EXTENSIONS:
            files.append(path)
    return files


def score_prd_file(path: Path):
    text = str(path).lower()
    score = 0
    if "prd" in text:
        score += 10
    if "master" in text or "masyer" in text:
        score += 10
    if "dataset" in text:
        score += 3
    if "kpi" in text:
        score += 2
    if "intent" in text:
        score += 2
    if "annual reports" in text or "brsr" in text or "esg dataset" in text:
        score -= 5
    return score


def find_prd_files(root: Path):
    files = table_files(root)
    scored = [(score_prd_file(path), path) for path in files]
    scored = [(score, path) for score, path in scored if score > 0]
    scored.sort(reverse=True, key=lambda item: (item[0], -len(str(item[1]))))
    return [path for _, path in scored]


PRD_FILES = find_prd_files(Path("/kaggle/input"))

if not PRD_FILES:
    print("No PRD master file found in /kaggle/input. Trying Google Drive folder...")
    drive_dir = OUTPUT_DIR / "drive_files"
    drive_dir.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download_folder(url=DRIVE_FOLDER_URL, output=str(drive_dir), quiet=False, use_cookies=False)
    except Exception as exc:
        print(f"Drive download failed or timed out: {exc}")
    PRD_FILES = find_prd_files(drive_dir)

if not PRD_FILES:
    all_found = table_files(Path("/kaggle/input")) + table_files(OUTPUT_DIR / "drive_files")
    print("Found table files, but none looked like PRD master:")
    for path in all_found[:50]:
        print("-", path)
    raise FileNotFoundError("No PRD master dataset found. Upload the PRD master file or folder as Kaggle input.")

print("Using PRD master file(s):")
for path in PRD_FILES:
    print("-", path)


# %% [markdown]
# # Cell 4: Load PRD master tables

# %%
def read_table(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file: {path}")


frames = []
source_counts = {}
for path in PRD_FILES:
    try:
        df = read_table(path)
    except Exception as exc:
        print(f"Skipping {path}: {exc}")
        continue
    df["__source_file"] = str(path)
    frames.append(df)
    source_counts[str(path)] = len(df)
    print(f"{path}: {len(df)} rows, {len(df.columns)} columns")

if not frames:
    raise RuntimeError("Could not load any PRD master table.")

prd_df = pd.concat(frames, ignore_index=True, sort=False)
prd_df = prd_df.drop_duplicates()
print(f"Combined PRD rows: {len(prd_df)}")
print("Columns:")
print(list(prd_df.columns))


# %% [markdown]
# # Cell 5: Build fine-tuning examples from PRD master

# %%
def clean_value(value):
    if pd.isna(value):
        return ""
    value = str(value)
    return re.sub(r"\s+", " ", value.replace("\ufeff", "")).strip()


def nonempty_ratio(series):
    values = series.map(clean_value)
    return (values != "").mean()


def avg_len(series):
    values = series.map(clean_value)
    nonempty = values[values != ""]
    if len(nonempty) == 0:
        return 0
    return nonempty.map(len).mean()


OUTPUT_CANDIDATE_NAMES = [
    "clean_esg_text",
    "cleaned_esg_text",
    "complete_esg_text",
    "extracted_esg_text",
    "esg_text",
    "report_text",
    "section_text",
    "paragraph_text",
    "generated_esg_text",
    "target_text",
    "output",
    "text",
    "llm_training_summary",
]


def choose_output_column(df):
    lower_to_col = {str(col).lower(): col for col in df.columns}
    for name in OUTPUT_CANDIDATE_NAMES:
        if name in lower_to_col:
            col = lower_to_col[name]
            if nonempty_ratio(df[col]) > 0.05 and avg_len(df[col]) > 80:
                return col

    scored = []
    for col in df.columns:
        lower = str(col).lower()
        if any(skip in lower for skip in ["evidence", "url", "file", "path", "source"]):
            continue
        textish = any(key in lower for key in ["text", "paragraph", "report", "summary", "output", "target"])
        if not textish:
            continue
        ratio = nonempty_ratio(df[col])
        length = avg_len(df[col])
        if ratio > 0.05 and length > 80:
            scored.append((length, ratio, col))
    if scored:
        scored.sort(reverse=True)
        return scored[0][2]

    print("Could not infer output column. Candidate long text columns:")
    for col in df.columns:
        length = avg_len(df[col])
        ratio = nonempty_ratio(df[col])
        if length > 40 and ratio > 0.05:
            print(f"- {col}: avg_len={length:.1f}, nonempty={ratio:.2%}")
    raise ValueError("Set OUTPUT_COLUMN manually in Cell 5.")


OUTPUT_COLUMN = choose_output_column(prd_df)
print(f"Using output column: {OUTPUT_COLUMN}")


INCLUDE_KEYS = [
    "company", "report", "year", "sector", "market", "framework", "assurance", "geography", "metadata", "meta",
    "kpi", "scope", "emission", "water", "waste", "energy", "renewable", "diversity", "women", "board",
    "net_zero", "target", "score", "has_", "classification", "class", "section", "topic", "category",
    "intent", "policy", "compliance", "risk", "governance", "operational", "forward", "count", "percent",
]

EXCLUDE_INPUT_KEYS = [
    "raw_text", "raw esg", "clean_text", "cleaned_text", "extracted_text", "complete_text", "paragraph_text",
    "report_text", "section_text", "generated", "summary", "output", "target", "evidence", "__source_file",
]


def include_input_column(col):
    if col == OUTPUT_COLUMN:
        return False
    lower = str(col).lower()
    if any(key in lower for key in EXCLUDE_INPUT_KEYS):
        return False
    return any(key in lower for key in INCLUDE_KEYS)


INPUT_COLUMNS = [col for col in prd_df.columns if include_input_column(col)]
if not INPUT_COLUMNS:
    INPUT_COLUMNS = [col for col in prd_df.columns if col != OUTPUT_COLUMN and col != "__source_file"]

print(f"Using {len(INPUT_COLUMNS)} input columns:")
print(INPUT_COLUMNS)


def make_input(row):
    lines = []
    for col in INPUT_COLUMNS:
        value = clean_value(row.get(col, ""))
        if value:
            label = str(col).replace("_", " ")
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


examples = []
seen = set()
for _, row in prd_df.iterrows():
    output = clean_value(row.get(OUTPUT_COLUMN, ""))
    input_text = make_input(row)
    if len(output) < 80 or len(input_text) < 20:
        continue
    example = {
        "instruction": (
            "Generate clean ESG section/report text from the provided KPI, metadata, "
            "ESG classification, and intent labels."
        ),
        "input": input_text,
        "output": output,
    }
    key = (example["instruction"], example["input"], example["output"])
    if key in seen:
        continue
    seen.add(key)
    examples.append(example)

random.seed(2024)
random.shuffle(examples)
if MAX_TRAIN_EXAMPLES:
    examples = examples[:MAX_TRAIN_EXAMPLES]

if not examples:
    raise RuntimeError("No training examples were created. Check OUTPUT_COLUMN and INPUT_COLUMNS.")

training_json = DATA_DIR / "prd_master_training.json"
manifest_json = DATA_DIR / "prd_master_training_manifest.json"
with training_json.open("w", encoding="utf-8") as handle:
    json.dump(examples, handle, indent=2, ensure_ascii=False)
with manifest_json.open("w", encoding="utf-8") as handle:
    json.dump(
        {
            "total_examples": len(examples),
            "output_column": str(OUTPUT_COLUMN),
            "input_columns": [str(col) for col in INPUT_COLUMNS],
            "source_counts": source_counts,
        },
        handle,
        indent=2,
        ensure_ascii=False,
    )

print(f"Prepared {len(examples)} training examples.")
print(f"Saved: {training_json}")
print(f"Manifest: {manifest_json}")
print(json.dumps(examples[0], indent=2, ensure_ascii=False)[:1800])


# %% [markdown]
# # Cell 6: Format dataset for Llama 3.1 chat fine-tuning

# %%
from datasets import Dataset


def to_llama_text(record):
    system = (
        "You are CarbonTatvaAI, an ESG reporting specialist. "
        "Your job is to generate clean ESG report text from structured KPI, metadata, classification, and intent data."
    )
    user = f"### Structured ESG Input:\n{record['input']}"
    return {
        "text": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{record['instruction']}\n\n{user}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{record['output']}<|eot_id|>"
        )
    }


dataset = Dataset.from_list([to_llama_text(item) for item in examples])
print(dataset)
print(dataset[0]["text"][:1000])


# %% [markdown]
# # Cell 7: Load model with Unsloth and attach adapters

# %%
from unsloth import FastLanguageModel


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=not USE_16BIT_LORA,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=2024,
    use_rslora=False,
    loftq_config=None,
)


# %% [markdown]
# # Cell 8: Train with 15-minute checkpoints

# %%
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

try:
    from unsloth.chat_templates import train_on_responses_only
except Exception:
    train_on_responses_only = None


class TimeBasedSaveCallback(TrainerCallback):
    def __init__(self, minutes=15):
        self.seconds = minutes * 60
        self.last_save_time = time.time()
        self.last_save_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        if state.global_step > self.last_save_step and now - self.last_save_time >= self.seconds:
            print(f"\nTime checkpoint: saving resumable checkpoint at step {state.global_step}")
            control.should_save = True
            self.last_save_time = now
            self.last_save_step = state.global_step
        return control


config_kwargs = {
    "output_dir": str(OUTPUT_DIR / "checkpoints"),
    "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "warmup_ratio": 0.05,
    "num_train_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "fp16": not torch.cuda.is_bf16_supported(),
    "bf16": torch.cuda.is_bf16_supported(),
    "logging_steps": 5,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "seed": 2024,
    "save_strategy": "steps",
    "save_steps": SAVE_STEPS,
    "save_total_limit": 3,
    "report_to": "none",
    "dataset_text_field": "text",
    "max_seq_length": MAX_SEQ_LENGTH,
    "max_length": MAX_SEQ_LENGTH,
    "dataset_num_proc": 2,
    "packing": False,
    "padding_free": False,
}
supported_config_args = set(inspect.signature(SFTConfig.__init__).parameters)
config_kwargs = {key: value for key, value in config_kwargs.items() if key in supported_config_args}
sft_config = SFTConfig(**config_kwargs)

trainer_kwargs = {
    "model": model,
    "tokenizer": tokenizer,
    "processing_class": tokenizer,
    "train_dataset": dataset,
    "args": sft_config,
}
supported_trainer_args = set(inspect.signature(SFTTrainer.__init__).parameters)
trainer_kwargs = {key: value for key, value in trainer_kwargs.items() if key in supported_trainer_args}
trainer = SFTTrainer(**trainer_kwargs)

if train_on_responses_only is not None:
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
else:
    print("Warning: train_on_responses_only unavailable. Training will use full formatted text.")

trainer.add_callback(TimeBasedSaveCallback(SAVE_EVERY_MINUTES))

checkpoint_root = OUTPUT_DIR / "checkpoints"
checkpoint_dirs = sorted(
    checkpoint_root.glob("checkpoint-*"),
    key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1,
)
resume_checkpoint = str(checkpoint_dirs[-1]) if checkpoint_dirs else None
if resume_checkpoint:
    print(f"Resuming from checkpoint: {resume_checkpoint}")

trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
print(trainer_stats)


# %% [markdown]
# # Cell 9: Save adapter

# %%
LORA_OUTPUT = OUTPUT_DIR / "final_lora_adapter"
model.save_pretrained(str(LORA_OUTPUT))
tokenizer.save_pretrained(str(LORA_OUTPUT))
print(f"Saved adapter to: {LORA_OUTPUT}")


# %% [markdown]
# # Cell 10: Test generation

# %%
FastLanguageModel.for_inference(model)

sample = examples[0]
test_prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are CarbonTatvaAI, an ESG reporting specialist. "
    "Your job is to generate clean ESG report text from structured KPI, metadata, classification, and intent data.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    f"{sample['instruction']}\n\n### Structured ESG Input:\n{sample['input']}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=700,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.08,
    do_sample=True,
)
generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

print("INPUT")
print(sample["input"][:1800])
print("\nGENERATED ESG TEXT")
print(generated)
print("\nREFERENCE ESG TEXT")
print(sample["output"][:2500])


# %% [markdown]
# # Cell 11: Copy outputs for download

# %%
kaggle_output = Path("/kaggle/working/carbontatva-prd-unsloth-lora-adapter")
shutil.copytree(LORA_OUTPUT, kaggle_output, dirs_exist_ok=True)

run_config = {
    "task": "prd_master_structured_esg_to_text",
    "model": MODEL_NAME,
    "training_method": TRAINING_METHOD,
    "training_mode": "16-bit LoRA" if USE_16BIT_LORA else "4-bit QLoRA",
    "examples": len(examples),
    "epochs": NUM_EPOCHS,
    "max_seq_length": MAX_SEQ_LENGTH,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "save_every_minutes": SAVE_EVERY_MINUTES,
    "output_column": str(OUTPUT_COLUMN),
    "input_columns": [str(col) for col in INPUT_COLUMNS],
    "adapter_output": str(kaggle_output),
}

with open("/kaggle/working/carbontatva_prd_unsloth_run_config.json", "w", encoding="utf-8") as handle:
    json.dump(run_config, handle, indent=2, ensure_ascii=False)

shutil.copytree(DATA_DIR, Path("/kaggle/working/carbontatva_prd_training_data"), dirs_exist_ok=True)

print("Done. Download from Kaggle Output:")
print(kaggle_output)
print("/kaggle/working/carbontatva_prd_unsloth_run_config.json")
print("/kaggle/working/carbontatva_prd_training_data")
print(json.dumps(run_config, indent=2))

gc.collect()
torch.cuda.empty_cache()
