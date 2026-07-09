# CarbonTatvaAI ESG Benchmarking Engine - Interview Story

This is not a script to memorize. Read it like a story. The goal is to help you explain the project naturally, with enough technical depth to guide the interviewer instead of being dragged by random questions.

## 1. The One-Minute Story

I worked on an ESG intelligence and benchmarking engine for Indian companies. The basic problem was that ESG/BRSR reports contain a lot of useful information, but it is scattered across PDFs, annual reports, BRSR disclosures, extracted KPI tables, and company-level metadata. If a company wants to know whether its disclosure is strong or weak, a single-company summary is not enough. It needs comparison.

So I built a pipeline that connects structured ESG/BRSR KPI data with available report evidence, then compares a selected company against either its sector or a manually selected peer group. The dashboard shows KPI values, peer median, peer average, peer range, rank, missing KPI opportunities, disclosure gaps, and report-source coverage.

In simple terms:

> I moved the project from "generate one ESG answer" to "benchmark a company against peers using structured ESG data and connected report evidence."

## 2. The Problem I Was Trying To Solve

Companies publish ESG/BRSR reports, but these reports are hard to compare.

For example, if a company reports Scope 1 emissions, the number alone does not tell much.

```text
Company A Scope 1 emissions = 10,000 tCO2e
```

This becomes useful only when compared:

```text
Company A: 10,000 tCO2e
Sector median: 14,000 tCO2e
Best peer: 7,000 tCO2e
Worst peer: 25,000 tCO2e
Rank: 8 out of 30
Interpretation: better than sector median
```

That is the benchmarking idea.

The project had three messy realities:

1. ESG information is spread across many files.
2. Raw PDFs are large and expensive to process every time.
3. A chatbot alone does not solve benchmarking, because benchmarking needs structured comparison, ranks, gaps, and peer statistics.

So the system needed a proper data layer before any LLM or chatbot layer.

## 3. The Key Insight

Initially, the direction looked like:

```text
Upload company report -> ask chatbot -> get ESG answer
```

But for the PRD, the better direction was:

```text
Company KPI + report evidence
        -> compare with sector/custom peers
        -> identify gaps
        -> generate insight/recommendation
```

This was the turning point.

A chatbot is useful later, but the core product is not just a chatbot. The core product is a benchmark engine.

## 4. What I Actually Built

I built a working V1 consisting of:

1. A report corpus connector
2. A benchmark company-year dataset
3. A sector/custom peer comparison engine
4. A static dashboard
5. A report-source coverage layer
6. A shareable hosted demo

The dashboard supports:

- target company selection
- reporting year selection
- sector-wise comparison
- custom peer-group comparison
- KPI comparison table
- missing KPI opportunity detection
- disclosure gap detection
- report-source coverage view
- visual comparison tab

This directly maps to the PRD idea of ESG intelligence and benchmarking.

## 5. Data I Used

### PRD/BRSR Master Dataset

This is the main structured data source.

It contains company-year-level ESG/BRSR information such as:

- company name
- reporting year
- sector
- ESG disclosure flags
- KPI values
- framework/assurance metadata
- available disclosure categories

This powers the actual comparison.

### BRSR 2021-24 Zip

I also used:

```text
C:\Users\Pratyush\Downloads\brsr 2021-24.zip
```

I did not extract all files manually. The connector indexes the zip members directly.

This gave:

```text
FY 2021-22: 177 BRSR PDFs
FY 2022-23: 801 BRSR PDFs
FY 2023-24: 1059 BRSR PDFs
```

### Scraped BRSR Reports For 2024-25 And 2025-26

These were already available locally from earlier scraping work:

```text
FY 2024-25: 2182 BRSR report entries
FY 2025-26: 192 BRSR report entries
```

### Annual Report Evidence

Raw annual reports are too large to download and process fully in a lightweight demo.

So I used already extracted annual-report evidence:

- annual-report KPI tables
- annual-report paragraph/intent extracted data

This gave:

```text
1402 annual-report extracted evidence rows
```

This is a practical engineering decision:

> Instead of carrying huge PDFs into the demo, use structured extracted evidence now, and keep raw annual PDF indexing as a future on-demand step.

## 6. Final Connected Corpus

The connected evidence corpus now has:

```text
5813 total report/evidence rows
4411 BRSR report entries
1402 annual-report extracted evidence entries
3864 benchmark company-year rows linked to evidence
```

This matters because the dashboard is not fake data. It is connected to real indexed ESG/BRSR artifacts.

## 7. How The Dashboard Works

The frontend is static HTML/CSS/JS. It does not need a backend server for the demo.

The pipeline exports benchmark data into:

```text
dashboard/dashboard_data.js
```

Then the dashboard reads this data in the browser and computes:

- target company value
- peer median
- peer average
- peer min/max range
- rank
- missing KPIs
- disclosure gaps
- report-source coverage

This makes the demo easy to share through GitHub Pages.

## 8. Sector-Wise Comparison

Sector-wise comparison means:

> Compare the selected company against companies from the same sector/year.

Example:

```text
Target: Aarti Drugs
Sector: Healthcare
Year: FY 2024-25
Output: Aarti Drugs vs Healthcare peers
```

This helps answer:

- Is the company better or worse than its sector?
- Which KPIs are sector-standard?
- Which disclosures are common in the sector?

## 9. Custom Peer Group Comparison

Custom peer group means:

> The user manually selects the companies to compare against.

Example:

```text
Target: TCS
Peers: Infosys, Wipro, HCL, Tech Mahindra
```

This is useful because sometimes the sector is too broad. A company may care about a specific peer set rather than the whole sector.

This was explicitly requested in the project discussion.

## 10. Missing KPI Opportunity

This means:

> A KPI is missing for the target company, but many peers disclose it.

Example:

```text
Target company does not disclose renewable energy share.
70% of selected peers disclose renewable energy share.
```

The system flags this as a reporting opportunity.

This is valuable because it helps companies improve future ESG/BRSR reporting.

## 11. Disclosure Gap

A disclosure gap means:

> The target company is not covering a topic that peers commonly cover.

Examples:

- climate risk
- Scope 3
- net-zero target
- diversity
- board governance
- water
- waste

This is different from a KPI gap.

KPI gap = missing number.

Disclosure gap = missing topic/section/coverage.

## 12. Report Sources Tab

The Report Sources tab shows:

- whether BRSR reports are connected
- whether annual evidence is connected
- how many peer companies have report evidence
- example source path

This makes the system more trustworthy because it shows the evidence coverage behind the benchmark.

## 13. Why This Is Better Than A Simple Chatbot

A chatbot can answer:

```text
What is the Scope 1 emission of Company A?
```

But benchmarking needs:

```text
Company A vs peer median
Company A rank
Missing KPIs
Peer disclosure adoption
Sector gap
Recommendation
```

That requires structured comparison logic.

So my position is:

> Chatbot can be the interaction layer later, but benchmark tables and peer comparison should be the intelligence layer underneath.

This is a strong point to say in interviews.

## 14. Where LLMs Fit In This Project

LLMs are useful, but they should not be used blindly.

In this project:

### Structured Benchmarking

Use deterministic logic:

- averages
- medians
- ranks
- gaps
- counts
- peer adoption rates

This should not be hallucinated by an LLM.

### RAG

Use RAG when:

- we need to answer from long reports
- we need report evidence
- we need citations/snippets
- we do not want to retrain the model

### Fine-Tuning

Use fine-tuning when:

- we want a model to follow a specific ESG reporting style
- we have many examples of KPI input -> ESG narrative output
- we want consistent section/report generation behavior

So the clean explanation is:

```text
Benchmarking logic = structured deterministic layer
RAG = evidence retrieval from reports
Fine-tuning = ESG writing style / report generation behavior
LLM = explanation and narrative layer
```

## 15. What I Tried Earlier

The project initially explored:

- SusGen-30K
- Llama 3.1 8B fine-tuning
- QLoRA/Unsloth
- KPI-to-ESG report generation
- RAG over company reports

This helped me understand the GenAI side.

But for the PRD, the more immediate deliverable became:

> a benchmark dashboard that compares companies and shows gaps.

So I shifted the work toward the actual product need.

This is a good story because it shows you did not just follow tools blindly; you adapted to the problem.

## 16. What Makes My Work Stand Out

You can say these points:

1. I did not stop at a static UI; I connected real benchmark data.
2. I added both sector-wise and custom peer-group comparison.
3. I connected BRSR report evidence across multiple years.
4. I indexed a 4.8 GB BRSR zip without extracting it.
5. I handled annual reports practically by using extracted annual evidence instead of downloading huge PDFs.
6. I separated structured benchmarking from LLM/RAG responsibilities.
7. I made the dashboard static and shareable through GitHub Pages.
8. I documented the full project story and data flow.

## 17. The Architecture In Simple Words

```text
PRD/BRSR master data
        +
BRSR PDFs / BRSR zip / annual extracted evidence
        |
        v
Corpus connector
        |
        v
Company-year benchmark table
        |
        v
Sector/custom peer comparison engine
        |
        v
Dashboard
        |
        v
KPI comparison + missing KPIs + disclosure gaps + evidence coverage
```

## 18. If Interviewer Asks: "What Was Your Role?"

Say:

> My role was to convert the project from loose ESG/LLM experimentation into a concrete benchmarking workflow. I worked on organizing the available ESG datasets, connecting BRSR/annual evidence sources, building comparison logic for company-vs-sector and company-vs-custom peers, and creating a dashboard that shows KPI comparisons, missing KPIs, disclosure gaps, and evidence coverage.

## 19. If Interviewer Asks: "Where Did You Use GenAI?"

Say:

> The broader project explored GenAI through Llama fine-tuning, SusGen-style ESG report generation, and RAG over company reports. But in the benchmarking engine, I intentionally kept the core comparison deterministic because ranks, medians, averages, and missing KPI checks should be reliable. GenAI can sit on top to generate explanations, but the benchmark numbers should come from structured data.

This is a mature answer.

## 20. If Interviewer Asks: "Why Not Just Use An LLM?"

Say:

> Because LLMs can hallucinate numbers. For ESG benchmarking, numerical fidelity matters. So I used structured data for KPI comparison and report-source indexing for evidence. The LLM layer can later summarize or explain the findings, but the underlying benchmark should be computed, not guessed.

## 21. If Interviewer Asks: "What Is RAG?"

Say:

> RAG means Retrieval-Augmented Generation. Instead of expecting the model to remember company-specific reports, we retrieve relevant report chunks at query time and pass them to the LLM as context. This gives more grounded answers and avoids retraining for every new report.

Then connect it to the project:

> In this project, RAG would be useful for answering questions from BRSR/annual reports, while the benchmark dashboard handles structured peer comparison.

## 22. If Interviewer Asks: "What Is Fine-Tuning?"

Say:

> Fine-tuning means training a base model on task-specific examples so it learns the desired behavior or output style. In our ESG use case, fine-tuning would be useful for converting KPI inputs into ESG narrative summaries or report sections.

Then add:

> But fine-tuning is not necessary for calculating peer medians or ranks. That should be deterministic.

## 23. If Interviewer Asks: "What Is QLoRA?"

Say:

> QLoRA is a parameter-efficient fine-tuning method. Instead of updating all model weights, it trains small adapter layers while loading the base model in 4-bit quantized form. This reduces GPU memory requirement and makes fine-tuning large models like Llama 3.1 8B possible on limited hardware.

## 24. If Interviewer Asks: "What Is The Product Value?"

Say:

> The product helps ESG teams see how their disclosures compare with peers before publishing. It can identify missing KPIs, weak disclosure areas, and peer standards. This can reduce manual benchmarking effort and improve report quality.

## 25. If Interviewer Asks: "What Was Difficult?"

Say:

> The main difficulty was not model training; it was data organization. ESG data existed in multiple forms: PRD master CSVs, BRSR PDFs, annual extracted KPIs, paragraph-intent files, and large zips. I had to connect these into a consistent company-year structure before any meaningful benchmarking could happen.

## 26. If Interviewer Asks: "What Would You Improve Next?"

Say:

1. Add better company-name matching using CIN/NSE/BSE identifiers.
2. Add evidence snippets from PDFs.
3. Add RAG for report-level question answering.
4. Add LLM-generated recommendations on top of deterministic benchmark outputs.
5. Add charts for trend analysis across years.
6. Add export to PDF/PowerPoint for ESG teams.
7. Add authentication and backend APIs for production.

## 27. How To Lead The Interview

Try to lead them like this:

> The interesting part was deciding what should be LLM-based and what should be deterministic. In ESG benchmarking, numbers should not be generated by an LLM. So I built the benchmark layer using structured KPI calculations, and kept LLM/RAG as a future explanation/evidence layer.

This sentence can steer the interviewer into a good technical discussion.

## 28. Good Papers To Read

You do not need to read all papers fully. Read abstracts, introduction, methodology diagrams, and conclusion.

### 1. SusGen-GPT

Title:

```text
SusGen-GPT: A Data-Centric LLM for Financial NLP and Sustainability Report Generation
```

Why it matters:

- closest to your original fine-tuning/RAG direction
- introduces SusGen-30K
- shows ESG report generation with smaller 7B/8B models
- combines data-centric training and RAG

Use it to say:

> This motivated the initial LLM fine-tuning and sustainability report-generation direction.

Link:

```text
https://arxiv.org/abs/2412.10906
```

### 2. ESGReveal

Title:

```text
ESGReveal: An LLM-based approach for extracting structured data from ESG reports
```

Why it matters:

- shows LLM + RAG for ESG report extraction
- relevant to extracting structured fields from messy reports
- supports your argument that ESG systems need retrieval and metadata, not just raw prompting

Use it to say:

> This paper supports the idea that ESG report intelligence needs extraction and retrieval pipelines.

Link:

```text
https://arxiv.org/abs/2312.17264
```

### 3. ESGBERT

Title:

```text
ESGBERT: Language Model to Help with Classification Tasks Related to Companies' Environmental, Social, and Governance Practices
```

Why it matters:

- older but foundational ESG-domain NLP paper
- shows domain-specific language models help ESG classification
- useful for explaining why general NLP models may not be enough for ESG text

Use it to say:

> Domain-specific ESG language has enough nuance that specialized models or domain adaptation can improve performance.

Link:

```text
https://arxiv.org/abs/2203.16788
```

### 4. ESG-CID

Title:

```text
ESG-CID: A Disclosure Content Index Finetuning Dataset for Mapping GRI and ESRS
```

Why it matters:

- relevant to retrieval and disclosure standards
- connects ESG report sections to standards like GRI/ESRS
- good future direction for benchmark/evidence retrieval

Use it to say:

> A future version can map company disclosures to reporting standards and improve retrieval quality.

Link:

```text
https://arxiv.org/abs/2503.10674
```

## 29. Your "I Stood Out" Version

Say this confidently:

> What I think I contributed was clarity. Initially, the project had many possible directions: fine-tuning, RAG, report generation, chatbot, scraping, and dashboards. I realized that for the PRD, the core need was benchmarking, not just generation. So I focused on building the structured comparison layer first: company vs sector/custom peers, KPI gaps, disclosure gaps, and evidence coverage. That gives a reliable base on which an LLM or chatbot can later sit.

This is probably your strongest story.

## 30. Final Short Version

If you only remember one flow, remember this:

```text
I worked on ESG benchmarking.
I connected PRD/BRSR structured data with BRSR and annual evidence.
I built company-vs-sector and company-vs-custom-peer comparison.
The dashboard shows KPI median/average/rank, missing KPIs, disclosure gaps, and report evidence.
I kept numerical benchmarking deterministic and positioned LLM/RAG as the explanation and evidence layer.
```

That is enough to sound clear and technically grounded.
