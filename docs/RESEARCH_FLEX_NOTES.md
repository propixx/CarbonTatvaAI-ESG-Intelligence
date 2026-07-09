# Research Flex Notes For ESG Benchmarking Interviews

This file is not for deep reading. It is for sounding informed.

Use these points when someone asks:

- Why this approach?
- Why not just use a chatbot?
- What papers did you look at?
- How is this different from generic GenAI?
- Why structured benchmarking before LLM?

## 1. The Main Flex Line

> I looked at ESG + LLM work and noticed that most systems fail when they treat ESG reporting as only a generation problem. ESG is evidence-heavy and comparison-heavy. So I designed the system with a deterministic benchmark layer first, and kept LLM/RAG as the explanation and evidence layer on top.

This is your main research-backed position.

## 2. The Stylish Reason Behind The Project

Most people think:

```text
ESG report + LLM = chatbot
```

But the better product thinking is:

```text
ESG report + peer data + structured KPIs = benchmark intelligence
```

So say:

> A chatbot can answer a question, but a benchmark engine can tell whether the answer is good or bad compared to the market.

That sounds strong.

## 3. Why Not Just Use An LLM?

Say:

> ESG benchmarking has numerical and compliance-sensitive parts. If the system has to calculate median, rank, peer range, and missing KPI coverage, I do not want the LLM to guess that. I want the numbers computed deterministically, and then an LLM can explain them.

Shorter:

> The LLM should narrate the benchmark, not invent the benchmark.

This is a very good line.

## 4. The Research Pattern You Noticed

Across ESG + LLM papers, the useful pattern is:

```text
Extract -> Structure -> Retrieve -> Compare -> Generate
```

Not:

```text
Dump PDF -> Ask LLM -> Trust answer
```

Say:

> The common pattern I saw in ESG LLM research is that raw generation alone is weak. Stronger systems first structure the data, retrieve evidence, and then generate grounded outputs.

## 5. Papers You Can Name Without Reading Fully

You only need to know how each paper supports your project direction.

## Paper 1: SusGen-GPT

Full name:

```text
SusGen-GPT: A Data-Centric LLM for Financial NLP and Sustainability Report Generation
```

What to say:

> SusGen-GPT influenced the initial thinking because it treats ESG report generation as a data-centric problem, not just prompting. It uses SusGen-30K and combines fine-tuning with RAG-style report generation.

How it connects to your work:

> This supported the idea that ESG models need domain-specific data and that report generation should be grounded in retrieved/company-specific information.

Stylish line:

> SusGen-GPT showed me that in ESG, data quality matters more than just model size.

Link:

```text
https://arxiv.org/abs/2412.10906
```

## Paper 2: ESGReveal

Full name:

```text
ESGReveal: An LLM-based approach for extracting structured data from ESG reports
```

What to say:

> ESGReveal was useful because it frames ESG work as extraction and structured analysis from reports, using LLMs with retrieval rather than simple free-form generation.

How it connects to your work:

> It supports my decision to connect report evidence and structured KPI data before generating insights.

Stylish line:

> ESGReveal reinforced that ESG intelligence starts with reliable extraction, not with pretty text generation.

Link:

```text
https://arxiv.org/abs/2312.17264
```

## Paper 3: ESGBERT

Full name:

```text
ESGBERT: Language Model to Help with Classification Tasks Related to Companies' Environmental, Social, and Governance Practices
```

What to say:

> ESGBERT is older but important because it shows that ESG language has domain-specific patterns, and generic language models may miss ESG-specific meaning.

How it connects to your work:

> It supports the idea that ESG systems need domain adaptation, taxonomy awareness, or structured labels instead of treating ESG text like normal text.

Stylish line:

> ESGBERT is a reminder that ESG is not just English text; it is domain language with financial and compliance context.

Link:

```text
https://arxiv.org/abs/2203.16788
```

## Paper 4: ESG-CID

Full name:

```text
ESG-CID: A Disclosure Content Index Finetuning Dataset for Mapping GRI and ESRS
```

What to say:

> ESG-CID is relevant because it focuses on mapping disclosure requirements to report content. That is close to the future version of our benchmark engine, where the system can map company disclosures to standards like GRI or ESRS.

How it connects to your work:

> It supports the future direction of standards-aware retrieval and gap analysis.

Stylish line:

> ESG-CID points toward standards-aware benchmarking, where we do not just compare companies, but compare them against disclosure frameworks.

Link:

```text
https://arxiv.org/abs/2503.10674
```

## 6. Your Research-Backed Design Decision

Say:

> After looking at these directions, I chose not to make the first version fully LLM-dependent. I made the benchmark layer structured and deterministic, because ESG benchmarking needs numerical reliability. LLM/RAG can come after that for evidence retrieval and natural-language explanation.

This makes you sound mature.

## 7. The "Three-Layer Architecture" Flex

Use this if someone asks architecture.

```text
Layer 1: Structured data layer
PRD/BRSR master data, KPIs, disclosure flags, company metadata

Layer 2: Benchmark intelligence layer
sector/custom peer selection, median, average, rank, missing KPIs, disclosure gaps

Layer 3: LLM/RAG narrative layer
evidence retrieval, explanation, report wording, recommendations
```

Say:

> I wanted the intelligence layer to be reliable before adding the language layer.

## 8. The "Why I Stood Out" Version

Say:

> My contribution was separating the product into the right layers. A lot of GenAI projects start by throwing documents into an LLM. I stepped back and asked what the business actually needs: peer comparison, gap detection, and evidence-backed recommendations. So I built the deterministic benchmarking base first.

## 9. If They Ask "What Research Did You Do?"

Say:

> I looked at four directions: ESG report generation, ESG data extraction, ESG-specific language models, and standards-aware retrieval. SusGen-GPT covered generation, ESGReveal covered extraction/RAG, ESGBERT covered domain-specific ESG NLP, and ESG-CID covered disclosure-standard mapping. Based on that, I decided our first version should be benchmark-first and LLM-second.

This is the cleanest answer.

## 10. If They Ask "Why Benchmarking?"

Say:

> ESG reports are useful only when they are comparable. A company knowing its Scope 1 emissions is not enough; it needs to know whether that number is above or below sector peers, what the peer median is, and whether it is missing disclosures that competitors already publish.

Shorter:

> ESG value comes from comparability, not just disclosure.

## 11. If They Ask "What Is Your Novelty?"

Do not claim research novelty. Claim engineering/product novelty.

Say:

> I would not call it research novelty; it is product engineering novelty. I connected scattered ESG/BRSR datasets, report evidence, and peer comparison into one usable dashboard. The value is in making ESG data comparable and actionable.

This is safe and smart.

## 12. If They Ask "Where Will LLM Come In?"

Say:

> Once the benchmark engine identifies gaps, an LLM can generate natural-language explanations: why the gap matters, which peers disclose it, and how the company can improve the section. But the LLM should consume benchmark outputs, not replace them.

Stylish line:

> The LLM is the analyst voice; the benchmark engine is the analyst spreadsheet.

## 13. If They Ask "Why RAG?"

Say:

> RAG is useful because company reports change every year. Instead of fine-tuning a model every time a new report comes, we can retrieve the relevant report evidence at query time and pass it to the LLM.

Shorter:

> RAG keeps the model grounded in the latest company documents.

## 14. If They Ask "Why Fine-Tuning?"

Say:

> Fine-tuning is useful if we want consistent ESG report-writing behavior, like KPI-to-narrative generation. But for benchmarking, fine-tuning is not the main requirement. Benchmarking needs structured comparison first.

Stylish line:

> Fine-tuning teaches style; benchmarking needs facts.

## 15. If They Ask "What Is The Future Direction?"

Say:

> The next step is to add evidence snippets and standards mapping. For example, if a company is missing Scope 3 disclosure, the system should show which peers disclose it, retrieve their relevant report snippets, and map the gap to BRSR/GRI/ESRS-style requirements.

This connects dashboard + RAG + standards.

## 16. Your Strongest 30-Second Answer

> I researched ESG LLM systems and noticed that the strong ones do not rely on raw generation. They structure data, retrieve evidence, and then generate. So for CarbonTatvaAI, I built the benchmark layer first: company vs sector/custom peers, KPI median/average/rank, missing KPI opportunities, disclosure gaps, and evidence coverage. LLM/RAG can later explain these findings, but the benchmark itself is deterministic because ESG numbers should not be hallucinated.

## 17. Your Strongest One-Liner

> In ESG, the LLM should explain the evidence, not become the evidence.

## 18. Another Good One-Liner

> A chatbot answers questions; a benchmark engine tells whether the answer is competitive.

## 19. Another Good One-Liner

> ESG reporting is not just generation, it is comparability plus evidence.

## 20. What To Avoid Saying

Do not say:

```text
I trained a model and it generates ESG reports perfectly.
```

Do not say:

```text
The LLM calculates all ranks and gaps.
```

Do not say:

```text
This is fully production-ready.
```

Say instead:

```text
This is a working V1 benchmark dashboard with real indexed data and a clear path to LLM/RAG-based explanations.
```

## 21. Final Flex Paragraph

> The main thing I learned is that ESG intelligence is not just an LLM problem. It is a data grounding problem. Reports are long, KPIs are scattered, company names are inconsistent, and comparison requires reliable statistics. So I built the foundation as a structured benchmark engine and used the research to decide where LLMs should fit: retrieval, explanation, and narrative generation, not raw numerical benchmarking.

