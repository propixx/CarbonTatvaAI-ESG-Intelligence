# CarbonTatvaAI Public Dataset Audit

Verified on 2026-05-27 from primary project/dataset pages where possible.

## Recommended Sources For Our Use Case

Our use case is: ESG/carbon disclosure assistant that can fine-tune on ESG instructions, retrieve evidence from company reports, and answer/report with grounded citations.

| Priority | Paper / Dataset | Public? | How We Use It | License / Risk |
|---|---:|---|---|---|
| 1 | SusGen-GPT / SusGen-30K | Yes | Core SFT data for financial + ESG instruction behavior | HuggingFace lists Apache-2.0. Good default. |
| 2 | Your BRSR/ESG CSV | Yes locally, because user supplied it | India/BRSR-specific carbon, KPI, and disclosure-summary examples | User-provided. Keep provenance. |
| 3 | ESG-CID | Yes | Retrieval/RAG training and evaluation for matching GRI/ESRS disclosure requirements to report chunks | HuggingFace lists CC-BY-NC-ND-4.0. Use for research/eval, avoid commercial derivative training unless cleared. |
| 4 | MMESGBench | Yes GitHub repo | RAG benchmark over ESG PDFs, especially visual/table/cross-page QA | Public repo, but no clear license found in page view. Use as evaluation until license is confirmed. |
| 5 | DynamicESG / ML-ESG tasks | Yes GitHub repo | Optional news impact/risk/opportunity classifier examples | GitHub lists CC-BY-SA-4.0. Share-alike implications. |
| 6 | ESG-Activities | Partly public GitHub files | Optional EU-taxonomy activity-detection examples | Public repo/files found, but no clear license found in page view. Use cautiously. |

## Not Added By Default

| Paper / Dataset | Reason |
|---|---|
| ESG-Bench | Paper says dataset exists, but I did not find a direct public dataset repo/HF download in the sources checked. Keep as method inspiration for hallucination evaluation. |
| ESGReveal | Strong method for RAG extraction from ESG reports, but no public training dataset found. We implement the method pattern: PDF parsing + retrieved context + structured ESG fields. |
| ESG-oriented LLM through ESG Practices | Useful LoRA/IRM method reference, but no public dataset located. |
| ESG-LLM scoring / ESGBERT / ESG rating papers | Often depend on proprietary ratings or filings datasets. Useful for architecture, not clean public training data here. |
| Healthcare Taiwan ESG report generation | Very aligned methodologically, but no public healthcare ESG training dataset found from the checked source. |
| Multimodal parsing papers like Pharos-ESG | Useful future direction, but heavier multimodal parsing is outside current Kaggle T4 text-only QLoRA pipeline. |

## Implementation Decision

Default training should stay:

```text
Llama 3.1 8B Instruct
+ QLoRA
+ SusGen-30K
+ user BRSR/ESG CSV
```

Optional research additions:

```text
+ ESG-CID for retrieval/ranking tasks
+ DynamicESG for risk/opportunity/duration tasks
+ ESG-Activities for EU taxonomy alignment
+ MMESGBench for RAG evaluation, not default SFT
```

This is safer than mixing every public-looking dataset into the adapter, because several useful ESG resources are non-commercial, share-alike, no-derivatives, or have unclear licenses.

