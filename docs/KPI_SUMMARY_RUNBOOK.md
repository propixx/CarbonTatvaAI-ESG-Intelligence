# CarbonTatvaAI KPI Summary Runbook

## What This Does

The model receives company metadata and structured BRSR KPIs and produces one
factual ESG narrative summary. The summary is a drafting aid, not a complete
statutory ESG or BRSR report.

The training data excludes SusGen, annual reports 2022-23, raw PDF text, intent
labels, section-classification labels, evidence text, and the old
`llm_training_summary` audit field.

## Files To Use

- `CarbonTatvaAI_KPI_Summary_Unsloth.ipynb`
- `CarbonTatvaAI_KPI_Summary_Dataset.zip`

The dataset bundle contains:

- `kpi_summary_train.jsonl`
- `kpi_summary_validation.jsonl`
- `kpi_summary_test.jsonl`
- `dataset_manifest.json`
- `dataset_quality.csv`
- `sample_summaries.json`
- `company_name_map.json`

## Kaggle Steps

1. Create a Kaggle notebook and enable a T4 GPU and Internet.
2. Add `CarbonTatvaAI_KPI_Summary_Dataset.zip` as notebook input and extract it,
   or add the extracted folder as a Kaggle dataset.
3. Import `CarbonTatvaAI_KPI_Summary_Unsloth.ipynb`.
4. Run the cells in order.
5. Keep `SMOKE_TEST = True` for a ten-step verification run.
6. After the smoke test succeeds, restart the session, set
   `SMOKE_TEST = False`, and run all cells for the one-epoch training run.
7. Run the Ollama export cell. It merges the trained LoRA adapter and creates a
   Q4_K_M GGUF model.
8. Download both:
   - `/kaggle/working/CarbonTatvaAI_KPI_Summary.zip`
   - `/kaggle/working/CarbonTatvaAI_Ollama.zip`

## Ollama Deployment

1. Install Ollama from <https://ollama.com/download>.
2. Extract `CarbonTatvaAI_Ollama.zip`.
3. Open PowerShell inside the extracted folder.
4. Create the local model:

   ```powershell
   ollama create carbontatva -f Modelfile
   ```

5. Test it:

   ```powershell
   ollama run carbontatva
   ```

6. Generate a report from structured KPI JSON:

   ```powershell
   python ollama_kpi_client.py example_input.json
   ```

Ollama serves the model locally through `http://localhost:11434/api`. This is a
local model/API deployment, not an automatically generated public website.

## Checkpoints

Training saves normal step checkpoints and also requests a checkpoint every
15 minutes. If Kaggle restarts, upload the latest `checkpoint-*` directory as a
Kaggle dataset. The training cell automatically imports the highest-step
checkpoint and resumes from it.

## Result To Present

> I prepared a company-level BRSR dataset where metadata and structured KPIs
> are the input and a grounded ESG narrative summary is the output. I
> fine-tuned Llama 3.1 8B using Unsloth QLoRA and evaluated numerical fidelity
> and KPI coverage on unseen companies.

Use `evaluation_metrics.json` and `heldout_comparisons.json` from the output
bundle to support the result.
