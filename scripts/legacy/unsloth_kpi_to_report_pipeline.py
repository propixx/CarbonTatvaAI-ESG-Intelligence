# =============================================================================
# CarbonTatvaAI - Unsloth KPI-to-ESG-Report Fine-Tuning Pipeline
# =============================================================================
# Core task:
#   input  = company name + ESG KPI data
#   output = ESG report-style narrative text
#
# Only company, disclosure, and KPI fields are used as input. The target is
# ESG report-style text.
#
# Default is QLoRA because it is the memory-efficient choice. If a premium GPU
# has enough VRAM and speed/quality matters more than memory, switch
# TRAINING_METHOD to "lora".
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
pip_install("gdown")

print("Dependencies installed.")


# %% [markdown]
# # Cell 2: Login, paths, and config

# %%
import csv
import gc
import json
import os
import random
import re
import time
from pathlib import Path

import torch

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OUTPUT_DIR = Path("/kaggle/working/carbontatva_unsloth")
DATA_DIR = OUTPUT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1HgnCVge9d9-EWDv9QbOiRMrFRylmPge5"

# Options:
#   "qlora" = 4-bit base model + LoRA adapters. More memory efficient.
#   "lora"  = 16-bit base model + LoRA adapters. More VRAM, sometimes faster.
TRAINING_METHOD = "qlora"

MODEL_NAME_16BIT = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME_4BIT = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

USE_16BIT_LORA = TRAINING_METHOD.lower() == "lora"
MODEL_NAME = MODEL_NAME_16BIT if USE_16BIT_LORA else MODEL_NAME_4BIT

MAX_SEQ_LENGTH = 768
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2 if USE_16BIT_LORA else 1
GRAD_ACCUM = 8 if USE_16BIT_LORA else 16
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
SAVE_STEPS = 50
SAVE_EVERY_MINUTES = 15

# For first smoke test, set this to 100 or 300.
# For full CSV training, keep None.
MAX_TRAIN_EXAMPLES = None

try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login

    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    login(token=hf_token)
    os.environ["HF_TOKEN"] = hf_token
    print("Logged into HuggingFace.")
except Exception as exc:
    print(f"HF_TOKEN Kaggle secret not found or login failed: {exc}")
    print("If running locally, login with: huggingface-cli login")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    raise RuntimeError("CUDA GPU required for this pipeline.")

print(f"Training mode: {'16-bit LoRA' if USE_16BIT_LORA else '4-bit QLoRA'}")
print(f"Model: {MODEL_NAME}")


# %% [markdown]
# # Cell 3: Find all CSV files

# %%
import gdown


def find_csvs(root: Path):
    if not root.exists():
        return []
    candidates = []
    for path in root.rglob("*.csv"):
        lowered = path.name.lower()
        score = sum(word in lowered for word in ["esg", "brsr", "prd", "master", "dataset"])
        candidates.append((score, path))
    if not candidates:
        return []
    candidates.sort(reverse=True, key=lambda item: item[0])
    return [path for _, path in candidates]


CSV_PATHS = find_csvs(Path("/kaggle/input"))
if not CSV_PATHS:
    print("No CSV found in /kaggle/input. Trying Google Drive folder...")
    drive_dir = OUTPUT_DIR / "drive_files"
    drive_dir.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download_folder(url=DRIVE_FOLDER_URL, output=str(drive_dir), quiet=False, use_cookies=False)
    except Exception as exc:
        print(f"Drive download failed: {exc}")
    CSV_PATHS = find_csvs(drive_dir)

if not CSV_PATHS:
    raise FileNotFoundError("No ESG/BRSR CSV found. Upload the CSV to Kaggle input or make the Drive folder public.")

print("Using CSV files:")
for path in CSV_PATHS:
    print(f"- {path}")


# %% [markdown]
# # Cell 4: Convert company + KPI rows to ESG report examples

# %%
META_FIELDS = [
    "company",
    "reporting_year",
    "meta_sector",
    "meta_market_cap",
    "meta_framework_used",
    "meta_brsr_version",
    "meta_assurance_type",
    "meta_geography",
]

DISCLOSURE_FLAG_FIELDS = [
    "has_environmental",
    "has_social",
    "has_governance",
    "has_climate_risk",
    "has_net_zero",
    "has_energy",
    "has_water",
    "has_waste",
    "has_scope_1",
    "has_scope_2",
    "has_scope_3",
    "has_diversity",
    "has_human_rights",
    "has_csr",
    "has_supply_chain",
    "has_board_governance",
    "has_tcfd",
    "has_ifrs_s1_s2",
    "has_cdp",
]

KPI_FIELDS = [
    "top_sections",
    "kpi_scope1_emissions_tco2e_current",
    "kpi_scope1_emissions_tco2e_previous",
    "kpi_scope1_emissions_yoy_reduction_percent",
    "kpi_scope2_emissions_tco2e_current",
    "kpi_scope2_emissions_tco2e_previous",
    "kpi_scope2_emissions_yoy_reduction_percent",
    "kpi_scope3_emissions_tco2e_current",
    "kpi_scope3_emissions_tco2e_previous",
    "kpi_scope3_emissions_yoy_reduction_percent",
    "kpi_scope1_scope2_total_tco2e_current",
    "kpi_scope1_scope2_total_tco2e_previous",
    "kpi_scope1_scope2_yoy_reduction_percent",
    "kpi_renewable_energy_percent",
    "kpi_renewable_energy_consumption_gj",
    "kpi_total_energy_consumption_gj",
    "kpi_water_consumption_kl_current",
    "kpi_water_consumption_kl_previous",
    "kpi_water_consumption_yoy_reduction_percent",
    "kpi_water_withdrawal_kl_current",
    "kpi_water_withdrawal_kl_previous",
    "kpi_water_withdrawal_yoy_reduction_percent",
    "kpi_total_waste_generated_current",
    "kpi_total_waste_generated_previous",
    "kpi_waste_recycled_current",
    "kpi_waste_recycled_previous",
    "kpi_waste_recycled_unit",
    "kpi_waste_recycled_percent",
    "kpi_female_employee_percent",
    "kpi_women_on_board_percent",
    "kpi_energy_intensity_current",
    "kpi_energy_intensity_previous",
    "kpi_energy_intensity_unit",
    "kpi_energy_intensity_yoy_reduction_percent",
    "kpi_net_zero_target_year",
    "kpi_targets_count",
    "kpi_direct_yoy_reductions_count",
]


def clean(value):
    value = "" if value is None else str(value)
    return re.sub(r"\s+", " ", value.replace("\ufeff", "")).strip()


def has_value(value):
    return clean(value) not in {"", "nan", "NaN", "None", "null", "[]", "{}"}


def label(field):
    return field.replace("meta_", "").replace("kpi_", "").replace("has_", "").replace("_", " ")


def build_kpi_input(row):
    lines = []
    for field in META_FIELDS + DISCLOSURE_FLAG_FIELDS + KPI_FIELDS:
        value = clean(row.get(field))
        if has_value(value):
            lines.append(f"{label(field)}: {value}")
    return "\n".join(lines)


def make_example(row):
    report_text = clean(row.get("llm_training_summary"))
    if not report_text:
        return None
    year = clean(row.get("reporting_year"))
    instruction = (
        "Generate a professional ESG report narrative from the provided company name and KPI data. "
        "Write coherent report-style disclosure text, not a summary and not bullet-point KPI extraction."
    )
    if year:
        instruction += f" Use the reporting period {year} where relevant."
    return {
        "instruction": instruction,
        "input": build_kpi_input(row),
        "output": report_text,
    }


examples = []
source_counts = {}
seen = set()
for csv_path in CSV_PATHS:
    count = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            example = make_example(row)
            if not example:
                continue
            key = (example["instruction"], example["input"], example["output"])
            if key in seen:
                continue
            seen.add(key)
            examples.append(example)
            count += 1
    source_counts[str(csv_path)] = count
    print(f"{csv_path}: {count} examples")

random.seed(2024)
random.shuffle(examples)
if MAX_TRAIN_EXAMPLES:
    examples = examples[:MAX_TRAIN_EXAMPLES]

dataset_path = DATA_DIR / "kpi_to_esg_report.json"
with dataset_path.open("w", encoding="utf-8") as handle:
    json.dump(examples, handle, indent=2, ensure_ascii=False)

manifest_path = DATA_DIR / "kpi_to_esg_report_manifest.json"
with manifest_path.open("w", encoding="utf-8") as handle:
    json.dump({"total_examples": len(examples), "sources": source_counts}, handle, indent=2, ensure_ascii=False)

print(f"Prepared {len(examples)} KPI-to-report examples.")
print(f"Manifest saved to {manifest_path}")
print(json.dumps(examples[0], indent=2, ensure_ascii=False)[:1200])


# %% [markdown]
# # Cell 5: Format dataset for Llama 3.1 chat fine-tuning

# %%
from datasets import Dataset


def to_llama_text(record):
    system = (
        "You are CarbonTatvaAI, an ESG reporting specialist. "
        "Your job is to transform company KPI data into professional ESG report narrative text."
    )
    user = f"### Company and KPI Data:\n{record['input']}"
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


# %% [markdown]
# # Cell 6: Load model with Unsloth and attach LoRA adapters

# %%
from unsloth import FastLanguageModel


dtype = None
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=dtype,
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
# # Cell 7: Train with 15-minute checkpoints

# %%
import inspect

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
    "dataset_num_proc": 2,
    "packing": False,
}
supported_config_args = set(inspect.signature(SFTConfig.__init__).parameters)
config_kwargs = {key: value for key, value in config_kwargs.items() if key in supported_config_args}
sft_config = SFTConfig(**config_kwargs)

trainer_kwargs = {
    "model": model,
    "tokenizer": tokenizer,
    "processing_class": tokenizer,
    "train_dataset": dataset,
    "dataset_text_field": "text",
    "max_seq_length": MAX_SEQ_LENGTH,
    "dataset_num_proc": 2,
    "packing": False,
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
# # Cell 8: Save LoRA adapter

# %%
LORA_OUTPUT = OUTPUT_DIR / "final_lora_adapter"
model.save_pretrained(str(LORA_OUTPUT))
tokenizer.save_pretrained(str(LORA_OUTPUT))
print(f"Saved LoRA adapter to: {LORA_OUTPUT}")


# %% [markdown]
# # Cell 9: Test generation

# %%
FastLanguageModel.for_inference(model)

sample = examples[0]
test_prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are CarbonTatvaAI, an ESG reporting specialist. "
    "Your job is to transform company KPI data into professional ESG report narrative text.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    f"{sample['instruction']}\n\n### Company and KPI Data:\n{sample['input']}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=700,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.08,
    do_sample=True,
)
generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print("INPUT KPI DATA")
print(sample["input"][:1500])
print("\nGENERATED ESG REPORT TEXT")
print(generated)
print("\nREFERENCE REPORT TEXT")
print(sample["output"][:2500])


# %% [markdown]
# # Cell 10: Copy outputs for Kaggle download

# %%
import shutil

kaggle_output = Path("/kaggle/working/carbontatva-unsloth-lora-adapter")
shutil.copytree(LORA_OUTPUT, kaggle_output, dirs_exist_ok=True)

run_config = {
    "task": "company_kpi_to_esg_report_text",
    "model": MODEL_NAME,
    "training_mode": "16-bit LoRA" if USE_16BIT_LORA else "4-bit QLoRA",
    "examples": len(examples),
    "source_counts": source_counts,
    "epochs": NUM_EPOCHS,
    "max_seq_length": MAX_SEQ_LENGTH,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "save_every_minutes": SAVE_EVERY_MINUTES,
    "output": str(kaggle_output),
}
with open("/kaggle/working/carbontatva_unsloth_run_config.json", "w", encoding="utf-8") as handle:
    json.dump(run_config, handle, indent=2)

print("Done. Download these from Kaggle Output:")
print(kaggle_output)
print("/kaggle/working/carbontatva_unsloth_run_config.json")
print(json.dumps(run_config, indent=2))

gc.collect()
torch.cuda.empty_cache()
