# CarbonTatvaAI Enterprise ESG Chatbot Runbook

## What This Notebook Does

This is the final chatbot-oriented deliverable for:

**ESG Chatbot for Enterprises - Snehashish + Pratyush**

It lets an enterprise user ask ESG questions such as:

- How do peers disclose Scope 3?
- Generate a board oversight disclosure.
- What does BRSR require for energy?
- Compare us against ITC and Tata Steel.
- Draft CBAM-aligned disclosures.

The notebook retrieves evidence from uploaded PDF, TXT, CSV, JSON, JSONL, and Excel files. It calls the local/public Ollama model when available. If Ollama is unavailable, it uses a clearly marked mock fallback so the demo still runs.

## How To Run

1. Open `CarbonTatvaAI_Enterprise_ESG_Chatbot_Final.ipynb` in Kaggle, Colab, or Jupyter.
2. Upload ESG reports, extracted ESG text, KPI tables, or PRD/BRSR tables as input files.
3. Run cells in order.
4. If Ollama is running, set:

   ```python
   OLLAMA_URL = "http://localhost:11434/api/generate"
   OLLAMA_MODEL = "carbontatva"
   ```

5. If using a temporary public Ollama endpoint, set:

   ```python
   OLLAMA_URL = "https://YOUR-TUNNEL-URL/api/generate"
   OLLAMA_MODEL = "carbontatva"
   ```

6. Run the example questions cell and the custom question cell.

## Stitch Integration

The notebook saves `stitch_frontend_api_contract.json`.

Use this contract for the Stitch UI:

- User enters a chatbot question.
- Frontend sends question to backend.
- Backend calls the notebook logic or Ollama API.
- Frontend displays answer and retrieved sources.

## Demo Statement

> I built the Enterprise ESG Chatbot flow. It ingests ESG reports or KPI files, retrieves relevant evidence, and answers enterprise ESG questions like Scope 3 peer disclosures, BRSR energy requirements, board oversight drafting, peer comparison, and CBAM-aligned disclosure drafting. It can use our Ollama model when available, with a marked mock fallback for demo continuity.

