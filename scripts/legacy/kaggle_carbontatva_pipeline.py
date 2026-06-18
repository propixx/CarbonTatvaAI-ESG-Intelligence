# =============================================================================
# CarbonTatvaAI - Kaggle QLoRA Pipeline
# =============================================================================
# How to use:
# 1. Kaggle -> New Notebook
# 2. Settings -> Accelerator -> GPU T4 x2 or P100
# 3. Settings -> Internet -> ON
# 4. Settings -> Add Secret -> Name: HF_TOKEN, Value: your HuggingFace token
# 5. Make sure your HF account has accepted:
#    https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# 6. Either upload the CSV as Kaggle data OR make the Drive folder public.
# 7. Copy this whole file into the notebook and run all cells.
#
# Main output:
# /kaggle/working/carbontatva/final_lora_adapter
# /kaggle/working/carbontatva-lora-adapter
# =============================================================================


# %% [markdown]
# # Cell 1: Install dependencies

# %%
import subprocess
import sys


def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


# Keep Kaggle's CUDA-enabled torch. Avoid reinstalling torch by accident.
pip_install("--upgrade", "--no-deps", "bitsandbytes>=0.43.3")
pip_install(
    "--upgrade",
    "transformers>=4.43.0",
    "accelerate>=0.33.0",
    "peft>=0.11.1",
    "trl>=0.9.6",
    "datasets>=2.20.0",
    "huggingface_hub",
    "sentencepiece",
    "protobuf",
    "bert-score",
    "rouge-score",
    "nltk",
    "sentence-transformers",
    "pypdf",
    "gdown",
)

print("Dependencies installed.")


# %% [markdown]
# # Cell 2: Login, paths, and GPU check

# %%
import gc
import json
import os
import random
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Simpler and more stable than letting device_map spread the model over 2 T4s.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

OUTPUT_DIR = Path("/kaggle/working/carbontatva")
DATA_DIR = OUTPUT_DIR / "data"
REPORTS_DIR = OUTPUT_DIR / "company_reports"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1HgnCVge9d9-EWDv9QbOiRMrFRylmPge5"

# Core project task:
#   input  = company name + ESG KPI data
#   output = ESG report-style narrative text
#
# Keep SusGen downloaded for later experiments, but do not mix it into the
# default training set because SusGen contains many ESG/finance tasks beyond
# KPI-to-report generation.
TASK_MODE = "kpi_to_report"
INCLUDE_SUSGEN_IN_TRAINING = False

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

try:
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    login(token=hf_token)
    print("Logged into HuggingFace.")
except Exception as exc:
    print(f"Could not read HF_TOKEN Kaggle secret: {exc}")
    print("Create a Kaggle secret named HF_TOKEN, then rerun.")
    raise

import torch

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected. Enable Kaggle GPU accelerator first.")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")


# %% [markdown]
# # Cell 3: Download SusGen-30K

# %%
from datasets import load_dataset


def load_susgen_dataset():
    for repo_id in ["WHATX/SusGen-30k", "WHATX/SusGen-30K"]:
        try:
            print(f"Trying {repo_id}...")
            return load_dataset(repo_id)
        except Exception as exc:
            print(f"Failed {repo_id}: {exc}")
    raise RuntimeError("Could not download SusGen-30K.")


print("Downloading SusGen-30K from HuggingFace...")
dataset = load_susgen_dataset()
split = dataset["train"] if "train" in dataset else dataset[next(iter(dataset.keys()))]

susgen_data = [
    {
        "instruction": str(item.get("instruction", "")),
        "input": str(item.get("input", "")),
        "output": str(item.get("output", "")),
    }
    for item in split
]

susgen_path = DATA_DIR / "susgen_30k.json"
with susgen_path.open("w", encoding="utf-8") as handle:
    json.dump(susgen_data, handle, indent=2, ensure_ascii=False)

print(f"Saved {len(susgen_data)} SusGen examples to {susgen_path}")


# %% [markdown]
# # Cell 4: Find or download your ESG CSV

# %%
import gdown


def find_csv(root: Path):
    if not root.exists():
        return None
    candidates = []
    for path in root.rglob("*.csv"):
        lowered = path.name.lower()
        score = 0
        for word in ["esg", "brsr", "prd", "master", "dataset"]:
            if word in lowered:
                score += 1
        candidates.append((score, path))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]


CSV_PATH = find_csv(Path("/kaggle/input"))

if CSV_PATH is None:
    print("No CSV found in /kaggle/input. Trying Google Drive folder...")
    drive_dir = OUTPUT_DIR / "drive_files"
    drive_dir.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download_folder(url=DRIVE_FOLDER_URL, output=str(drive_dir), quiet=False, use_cookies=False)
    except Exception as exc:
        print(f"Drive download failed: {exc}")
    CSV_PATH = find_csv(drive_dir)

if CSV_PATH:
    print(f"Using CSV: {CSV_PATH}")
else:
    print("No ESG CSV found. Training will use SusGen-30K only.")


# %% [markdown]
# # Cell 5: Convert ESG CSV into better instruction examples

# %%
BOOL_FIELDS = [
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
    "kpi_total_energy_consumption_gj",
    "kpi_water_consumption_kl_current",
    "kpi_total_waste_generated_current",
    "kpi_waste_recycled_percent",
    "kpi_female_employee_percent",
    "kpi_women_on_board_percent",
    "kpi_energy_intensity_current",
    "kpi_net_zero_target_year",
]

EVIDENCE_FIELDS = [
    "kpi_scope1_evidence",
    "kpi_scope2_evidence",
    "kpi_scope3_evidence",
    "kpi_renewable_energy_evidence",
    "kpi_water_consumption_evidence",
    "kpi_waste_recycled_evidence",
    "kpi_total_waste_generated_evidence",
    "kpi_female_employee_evidence",
    "kpi_energy_intensity_evidence",
]


def clean(value):
    value = "" if value is None else str(value)
    return re.sub(r"\s+", " ", value.replace("\ufeff", "")).strip()


def has_value(value):
    return clean(value) not in {"", "nan", "NaN", "None", "null"}


def label(field):
    return field.replace("kpi_", "").replace("has_", "").replace("_", " ")


def truthy(value):
    return clean(value).lower() in {"true", "1", "yes", "y"}


def fmt_num(value, suffix=""):
    value = clean(value)
    if not value:
        return "not disclosed"
    try:
        number = float(value.replace(",", ""))
        return f"{number:,.2f}".rstrip("0").rstrip(".") + suffix
    except ValueError:
        return value


def truncate(value, limit=900):
    value = clean(value)
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def base_context(row):
    present = [label(field) for field in BOOL_FIELDS if truthy(row.get(field))]
    absent = [label(field) for field in BOOL_FIELDS if clean(row.get(field)) and not truthy(row.get(field))]
    parts = [
        f"Company: {clean(row.get('company'))}",
        f"Reporting year: {clean(row.get('reporting_year'))}",
        f"Sector: {clean(row.get('meta_sector'))}",
        f"Market cap: {clean(row.get('meta_market_cap'))}",
        f"Framework: {clean(row.get('meta_framework_used'))}",
        f"Assurance: {clean(row.get('meta_assurance_type'))}",
        f"Geography: {clean(row.get('meta_geography'))}",
        f"Top sections: {clean(row.get('top_sections'))}",
        f"Present disclosure areas: {', '.join(present) if present else 'not identified'}",
        f"Missing disclosure areas: {', '.join(absent) if absent else 'none identified'}",
    ]
    kpi_lines = [f"- {label(field)}: {clean(row.get(field))}" for field in KPI_FIELDS if has_value(row.get(field))]
    if kpi_lines:
        parts.append("KPIs:\n" + "\n".join(kpi_lines))
    evidence_lines = [
        f"- {label(field)}: {truncate(row.get(field), 500)}"
        for field in EVIDENCE_FIELDS
        if has_value(row.get(field))
    ]
    if evidence_lines:
        parts.append("Evidence:\n" + "\n".join(evidence_lines))
    return "\n".join(part for part in parts if part.strip() and not part.endswith(": "))


def kpi_report_context(row):
    fields = [
        "company",
        "reporting_year",
        "meta_sector",
        "meta_market_cap",
        "meta_framework_used",
        "meta_brsr_version",
        "meta_assurance_type",
        "meta_geography",
    ] + BOOL_FIELDS + KPI_FIELDS
    lines = []
    for field in fields:
        value = clean(row.get(field))
        if has_value(value):
            lines.append(f"{label(field)}: {value}")
    return "\n".join(lines)


def carbon_output(row):
    company = clean(row.get("company"))
    year = clean(row.get("reporting_year"))
    lines = [
        f"For {company} in {year}, the available GHG emissions data indicates:",
        f"- Scope 1 emissions: {fmt_num(row.get('kpi_scope1_emissions_tco2e_current'), ' tCO2e')} current versus {fmt_num(row.get('kpi_scope1_emissions_tco2e_previous'), ' tCO2e')} previous, with YoY reduction of {fmt_num(row.get('kpi_scope1_emissions_yoy_reduction_percent'), '%')}.",
        f"- Scope 2 emissions: {fmt_num(row.get('kpi_scope2_emissions_tco2e_current'), ' tCO2e')} current versus {fmt_num(row.get('kpi_scope2_emissions_tco2e_previous'), ' tCO2e')} previous, with YoY reduction of {fmt_num(row.get('kpi_scope2_emissions_yoy_reduction_percent'), '%')}.",
    ]
    if has_value(row.get("kpi_scope3_emissions_tco2e_current")):
        lines.append(
            f"- Scope 3 emissions: {fmt_num(row.get('kpi_scope3_emissions_tco2e_current'), ' tCO2e')} current versus {fmt_num(row.get('kpi_scope3_emissions_tco2e_previous'), ' tCO2e')} previous, with YoY reduction of {fmt_num(row.get('kpi_scope3_emissions_yoy_reduction_percent'), '%')}."
        )
    if has_value(row.get("kpi_scope1_scope2_total_tco2e_current")):
        lines.append(
            f"- Combined Scope 1 and Scope 2 emissions: {fmt_num(row.get('kpi_scope1_scope2_total_tco2e_current'), ' tCO2e')} current versus {fmt_num(row.get('kpi_scope1_scope2_total_tco2e_previous'), ' tCO2e')} previous, with YoY reduction of {fmt_num(row.get('kpi_scope1_scope2_yoy_reduction_percent'), '%')}."
        )
    return "\n".join(lines)


def coverage_output(row):
    present = [label(field) for field in BOOL_FIELDS if truthy(row.get(field))]
    absent = [label(field) for field in BOOL_FIELDS if clean(row.get(field)) and not truthy(row.get(field))]
    return (
        f"The disclosure for {clean(row.get('company'))} covers {len(present)} ESG topic areas.\n"
        f"Covered areas include: {', '.join(present) if present else 'none identified'}.\n"
        f"Gaps or absent areas include: {', '.join(absent) if absent else 'none identified'}."
    )


esg_examples = []

if CSV_PATH:
    import csv

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            context = base_context(row)
            summary = clean(row.get("llm_training_summary"))
            if not summary:
                summary = (
                    f"{clean(row.get('company'))} reported for {clean(row.get('reporting_year'))}. "
                    "The available data summarizes the company's ESG coverage, KPIs, and disclosure gaps."
                )
            if TASK_MODE == "kpi_to_report":
                esg_examples.append(
                    {
                        "instruction": (
                            "Generate a professional ESG report narrative from the provided company name "
                            "and KPI data. Write report-style disclosure text, not a bullet summary."
                        ),
                        "input": kpi_report_context(row),
                        "output": summary,
                    }
                )
            else:
                esg_examples.append(
                    {
                        "instruction": "Prepare a structured ESG disclosure summary for this company.",
                        "input": context,
                        "output": summary,
                    }
                )
                if has_value(row.get("kpi_scope1_emissions_tco2e_current")) or has_value(row.get("kpi_scope2_emissions_tco2e_current")):
                    esg_examples.append(
                        {
                            "instruction": "Summarize the company's greenhouse gas emissions and year-on-year changes.",
                            "input": context,
                            "output": carbon_output(row),
                        }
                    )
                esg_examples.append(
                    {
                        "instruction": "Identify the ESG disclosure coverage and important reporting gaps.",
                        "input": context,
                        "output": coverage_output(row),
                    }
                )

esg_path = DATA_DIR / "esg_csv_instruction.json"
with esg_path.open("w", encoding="utf-8") as handle:
    json.dump(esg_examples, handle, indent=2, ensure_ascii=False)

print(f"Generated {len(esg_examples)} CSV-derived ESG examples.")


# %% [markdown]
# # Cell 6: Merge training data

# %%
# Set to a number like 12000 for a faster first Kaggle smoke run.
# Set to None for full SusGen-30K + CSV.
MAX_TRAIN_EXAMPLES = None

all_data = (susgen_data if INCLUDE_SUSGEN_IN_TRAINING else []) + esg_examples
random.seed(2024)
random.shuffle(all_data)

if MAX_TRAIN_EXAMPLES:
    all_data = all_data[:MAX_TRAIN_EXAMPLES]

training_path = DATA_DIR / "carbontatva_training.json"
with training_path.open("w", encoding="utf-8") as handle:
    json.dump(all_data, handle, indent=2, ensure_ascii=False)

print(f"Final training examples: {len(all_data)}")
print(f"Saved to: {training_path}")


# %% [markdown]
# # Cell 7: Fine-tune Llama 3.1 8B with QLoRA

# %%
import time

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments


RESULTS_DIR = str(OUTPUT_DIR / "results")
LORA_OUTPUT = str(OUTPUT_DIR / "final_lora_adapter")

# T4-safe defaults. Full 3 epochs can exceed Kaggle session time.
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 384
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SAVE_EVERY_MINUTES = 15


def make_prompt_and_answer(record):
    system_msg = (
        "You are CarbonTatvaAI, an expert ESG and sustainability report analyst. "
        "Provide detailed, accurate, and well-structured responses about carbon, climate risk, and ESG disclosures."
    )
    instruction = str(record.get("instruction", "")).strip()
    user_input = str(record.get("input", "")).strip()
    output = str(record.get("output", "")).strip()
    user_msg = instruction if not user_input else f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}"
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    answer = f"{output}<|eot_id|>"
    return {"prompt": prompt, "answer": answer}


with training_path.open("r", encoding="utf-8") as handle:
    raw_data = json.load(handle)

dataset = Dataset.from_list([make_prompt_and_answer(item) for item in raw_data])
print(f"Loaded {len(dataset)} formatted examples.")

use_bf16 = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
print(f"Using {'BF16' if use_bf16 else 'FP16'} compute dtype.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=compute_dtype,
    trust_remote_code=True,
)
model.config.use_cache = False

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


def tokenize_example(example):
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(example["answer"], add_special_tokens=False)["input_ids"]

    # Keep at least 64 tokens for the answer whenever possible.
    max_prompt = MAX_SEQ_LENGTH - min(64, len(answer_ids))
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[:max_prompt]

    input_ids = (prompt_ids + answer_ids)[:MAX_SEQ_LENGTH]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_SEQ_LENGTH]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


tokenized_dataset = dataset.map(tokenize_example, remove_columns=["prompt", "answer"])
print(f"Tokenized {len(tokenized_dataset)} examples.")


def data_collator(features):
    input_ids = [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features]
    labels = [torch.tensor(feature["labels"], dtype=torch.long) for feature in features]
    attention_mask = [torch.tensor(feature["attention_mask"], dtype=torch.long) for feature in features]
    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
    }


training_kwargs = {
    "output_dir": RESULTS_DIR,
    "num_train_epochs": NUM_EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "learning_rate": LEARNING_RATE,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "fp16": not use_bf16,
    "bf16": use_bf16,
    "logging_steps": 25,
    "save_steps": 500,
    "save_total_limit": 2,
    "optim": "paged_adamw_32bit",
    "max_grad_norm": 0.3,
    "group_by_length": True,
    "report_to": "none",
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
}

# Kaggle images sometimes carry older/newer transformers builds. Filter unsupported
# args instead of crashing on harmless compatibility differences.
import inspect

supported_training_args = set(inspect.signature(TrainingArguments.__init__).parameters)
training_kwargs = {key: value for key, value in training_kwargs.items() if key in supported_training_args}
training_args = TrainingArguments(**training_kwargs)


class TimeBasedSaveCallback(TrainerCallback):
    def __init__(self, minutes):
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


trainer_kwargs = {
    "model": model,
    "train_dataset": tokenized_dataset,
    "tokenizer": tokenizer,
    "processing_class": tokenizer,
    "args": training_args,
    "data_collator": data_collator,
    "callbacks": [TimeBasedSaveCallback(SAVE_EVERY_MINUTES)],
}
supported_trainer_args = set(inspect.signature(Trainer.__init__).parameters)
trainer_kwargs = {key: value for key, value in trainer_kwargs.items() if key in supported_trainer_args}
trainer = Trainer(**trainer_kwargs)

checkpoint_dirs = sorted(Path(RESULTS_DIR).glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
resume_checkpoint = str(checkpoint_dirs[-1]) if checkpoint_dirs else None
if resume_checkpoint:
    print(f"Resuming from checkpoint: {resume_checkpoint}")

print("Starting training.")
trainer.train(resume_from_checkpoint=resume_checkpoint)

trainer.model.save_pretrained(LORA_OUTPUT)
tokenizer.save_pretrained(LORA_OUTPUT)
print(f"Training complete. LoRA adapter saved to {LORA_OUTPUT}")

del trainer, model
torch.cuda.empty_cache()
gc.collect()


# %% [markdown]
# # Cell 8: Test the trained model

# %%
from peft import PeftModel

inference_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=inference_dtype,
)

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=inference_dtype,
)
model = PeftModel.from_pretrained(base, LORA_OUTPUT)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


def infer(instruction, text_input, max_new_tokens=512):
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
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


test_cases = [
    (
        "Summarize the greenhouse gas emissions disclosed in the following text.",
        "Tata Steel reported total GHG emissions of 30.5 million tonnes CO2e for FY2024. "
        "Scope 1 was 28.2 MT from blast furnaces. Scope 2 was 2.1 MT from electricity. "
        "Emission intensity was 2.07 tCO2e per tonne of crude steel, down 3.2% YoY.",
    ),
    (
        "Identify the key climate-related risks from this disclosure.",
        "The company's coastal plant faces flood risk from rising sea levels. "
        "EU carbon pricing could raise costs by 15-20%. Water scarcity has intensified.",
    ),
    (
        "Draft a TCFD-aligned governance disclosure.",
        "The Board oversees climate risks via its Sustainability Committee quarterly. "
        "CEO owns climate strategy. Net-zero by 2050, interim 30% reduction by 2030 from 2020 baseline.",
    ),
]

for idx, (instruction, text_input) in enumerate(test_cases, 1):
    print("\n" + "=" * 70)
    print(f"TEST {idx}: {instruction}")
    print("=" * 70)
    print(infer(instruction, text_input))


# %% [markdown]
# # Cell 9: Small evaluation

# %%
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

NUM_EVAL = 30
random.seed(2024)
eval_samples = random.sample(all_data, min(NUM_EVAL, len(all_data)))

predictions = []
references = []

for idx, sample in enumerate(eval_samples, 1):
    pred = infer(sample["instruction"], sample["input"], max_new_tokens=256)
    predictions.append(pred)
    references.append(sample["output"])
    if idx % 10 == 0:
        print(f"Generated {idx}/{len(eval_samples)}")

print("Computing metrics.")
_, _, f1 = bert_score_fn(predictions, references, lang="en", verbose=False, device="cpu")
bert_f1 = f1.mean().item()

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l = sum(scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(references, predictions)) / len(predictions)

smooth = SmoothingFunction().method1
bleu = sum(
    sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
    for ref, pred in zip(references, predictions)
) / len(predictions)

print("=" * 40)
print(f"EVALUATION RESULTS ({len(eval_samples)} samples)")
print("=" * 40)
print(f"BERTScore F1: {bert_f1:.4f}")
print(f"ROUGE-L:      {rouge_l:.4f}")
print(f"BLEU:         {bleu:.4f}")


# %% [markdown]
# # Cell 10: Save outputs

# %%
import shutil

kaggle_output = Path("/kaggle/working/carbontatva-lora-adapter")
if Path(LORA_OUTPUT).exists():
    shutil.copytree(LORA_OUTPUT, kaggle_output, dirs_exist_ok=True)
    print(f"LoRA adapter copied to {kaggle_output}")

results = {
    "model": MODEL_NAME,
    "training_examples": len(all_data),
    "epochs": NUM_EPOCHS,
    "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
    "learning_rate": LEARNING_RATE,
    "lora_r": LORA_R,
    "max_seq_length": MAX_SEQ_LENGTH,
    "bert_score_f1": round(bert_f1, 4),
    "rouge_l": round(rouge_l, 4),
    "bleu": round(bleu, 4),
    "adapter_path": str(kaggle_output),
}

with open("/kaggle/working/eval_results.json", "w", encoding="utf-8") as handle:
    json.dump(results, handle, indent=2)

print("Done. Download from Kaggle Output tab.")
print(json.dumps(results, indent=2))


# %% [markdown]
# # Cell 11 optional: Push adapter to HuggingFace Hub

# %%
# from huggingface_hub import HfApi
# api = HfApi()
# api.upload_folder(
#     folder_path=str(kaggle_output),
#     repo_id="YOUR_USERNAME/CarbonTatvaAI-LoRA",
#     repo_type="model",
#     create_repo=True,
# )
# print("Uploaded to HuggingFace.")
