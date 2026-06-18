# CarbonTatvaAI GPU Runbook

This repo has been prepared with the CarbonTatvaAI scripts, config, and the uploaded
ESG/BRSR CSV under `data/esg_prd_master_dataset_25-26.csv`.

## What Is Already Prepared

- `download_data.py` downloads `WHATX/SusGen-30K` from HuggingFace.
- `convert_esg_csv_to_susgen.py` converts the uploaded ESG CSV into instruction examples.
- `build_training_data.py` merges SusGen-30K with the uploaded ESG instruction data.
- `train_carbontatva.py` fine-tunes Llama 3.1 8B Instruct with QLoRA.
- `test_model.py` runs ESG/carbon sample inference.
- `rag_inference.py` runs RAG over PDFs/TXT files in `company_reports/`.
- `evaluate_model.py` computes BERTScore, ROUGE-L, and BLEU.
- `merge_model.py` merges the LoRA adapter into the base model for deployment.
- `src/configs/training_configs/carbontatva_config.yaml` is present for repo-style config runs.

## GPU Machine Commands

Run on Linux with an NVIDIA GPU and at least one accepted HuggingFace token for
`meta-llama/Llama-3.1-8B-Instruct`.

```bash
cd SusGen
bash setup_carbontatva_linux.sh
conda activate carbontatva
huggingface-cli login
export WANDB_DISABLED=true

python download_data.py
python convert_esg_csv_to_susgen.py
python build_training_data.py
python train_carbontatva.py
```

After training:

```bash
python test_model.py
python evaluate_model.py --num-samples 50
```

For RAG, place PDFs in `company_reports/`, then run:

```bash
python rag_inference.py
```

For deployment merge:

```bash
python merge_model.py
```

## Notes

- This local Windows runtime cannot run the CUDA training path because `nvidia-smi`
  and `conda` are not available here.
- If a 24 GB GPU runs out of memory, retry training with:

```bash
python train_carbontatva.py --batch-size 2 --grad-accum 8
```

- If the GPU does not support BF16, the training script automatically falls back to FP16.
