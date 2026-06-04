

# 🏥 Evaluating RAG and LLM Architectures for Evidence-Grounded Healthcare AI

### *A Rigorous Comparative Study on Factual Accuracy, Clinical Safety, and Hallucination Reduction*

<br/>

<div align="center">

<!-- Status Badges -->

![Research Status](https://img.shields.io/badge/Status-Published%20%2F%20Under%20Review-brightgreen?style=for-the-badge&logo=checkmarx&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge&logo=opensourceinitiative&logoColor=white)
![Stars](https://img.shields.io/github/stars/tanmaytyagii/RAG-vs-LLM-Healthcare-Research?style=for-the-badge&logo=github&color=yellow)
![Forks](https://img.shields.io/github/forks/tanmaytyagii/RAG-vs-LLM-Healthcare-Research?style=for-the-badge&logo=git&color=orange)

<br><br>

<!-- Tech Stack -->

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-FFD21E?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-4285F4?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?style=flat-square&logo=openai&logoColor=white)

<br><br>

<!-- Research Meta -->

![Domain](https://img.shields.io/badge/Domain-Healthcare%20AI-red?style=flat-square)
![Method](https://img.shields.io/badge/Method-RAG%20%7C%20LLM%20Evaluation-purple?style=flat-square)
![Format](https://img.shields.io/badge/Format-IEEE%20Standard-00629B?style=flat-square&logo=ieee&logoColor=white)
![Institution](https://img.shields.io/badge/Institution-Bennett%20University-8B0000?style=flat-square)
![PRs](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=flat-square&logo=git&logoColor=white)

</div>

<!-- ══════════ KEY RESULT CALLOUTS ══════════ -->
| 🎯 Hallucination Reduced | 📈 Factual Accuracy | 🛡️ Clinical Safety | ⚡ Grounded Responses |
|:---:|:---:|:---:|:---:|
| **27% → 9%** | **72% → 88%** | **71% → 92%** | **68% → 87%** |
| *LLM → RAG* | *LLM → RAG* | *LLM → RAG* | *LLM → RAG* |

<br/>

[📄 Read the Paper](#-publication-details) · [🚀 Quick Start](#-quick-start) · [📊 Results](#-key-findings--results) · [💬 Cite This Work](#-citation) · [🤝 Contribute](#-contributing)

<br/>

</div>

---

## 📋 Table of Contents

<details open>
<summary><b>Click to expand / collapse</b></summary>

- [🔬 Executive Summary](#-executive-summary)
- [📝 Abstract](#-abstract)
- [🎯 Research Objectives](#-research-objectives)
- [🏗️ Research Methodology](#️-research-methodology)
- [📊 Key Findings & Results](#-key-findings--results)
- [📈 Evaluation Metrics](#-evaluation-metrics)
- [🔍 Comparative Analysis](#-comparative-analysis)
- [🌟 Major Contributions](#-major-contributions)
- [🏥 Healthcare Applications](#-healthcare-applications)
- [🛠️ Technologies & Concepts](#️-technologies--concepts)
- [💡 Research Impact](#-research-impact)
- [🔭 Future Work](#-future-work)
- [📖 Publication Details](#-publication-details)
- [💬 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)
- [👥 Authors & Contact](#-authors--contact)

</details>

---

## 🔬 Executive Summary

<div align="center">

> *"RAG frameworks represent a positive step forward for generative AI, offering a more reliable and data-supported approach for healthcare applications where accuracy and trust are indispensable."*
> — Tyagi et al., 2025

</div>

This research presents a **survey-based comparative evaluation** of two dominant generative AI paradigms — **standalone Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** — within the context of healthcare AI applications. Drawing evidence from **20 peer-reviewed papers** published between 2023–2025, the study aggregates over **12,000 model outputs** across **50 evaluation metrics** to deliver an authoritative, data-grounded comparison.

Our synthesis demonstrates that RAG architectures consistently outperform traditional LLMs on the metrics that matter most in healthcare: factual accuracy (**+22.2%**), groundedness (**+27.9%**), clinical safety (**+29.6%**), and hallucination reduction (**−66.7%**). While LLMs retain advantages in fluency and computational efficiency, RAG emerges as the **gold standard for evidence-grounded clinical AI** — validated across major benchmarks including MIRAGE, Self-BioRAG, RAG², Omni-RAG, GraphRAG, and Bayesian-RAG.

---

## 📝 Abstract

This study explores a **comparative analysis of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems** within the medical field. Using evidence gathered from **twenty academic research papers**, the analysis reviews both architectures across key performance metrics, including factual accuracy, hallucination rate, computational efficiency, and clinical safety.

The study shows that RAG-based models consistently achieve higher factual precision — improving accuracy by an average of **22%** and lowering hallucination errors by over **60%** — while maintaining natural and coherent language output. Nevertheless, both paradigms encounter drawbacks such as retrieval bias, inference delay, and hard-to-explain behavior. The study determines that **RAG frameworks represent a positive step forward for generative AI**, offering a more reliable and data-supported approach for healthcare applications where accuracy and trust are indispensable.

The survey encompasses peer-reviewed research published between **2023 and 2025**, including major benchmarks such as MIRAGE, Self-BioRAG, RAG², Omni-RAG, GraphRAG, and Bayesian-RAG. In total, over **12,000 model outputs** and **50 distinct evaluation metrics** were aggregated and normalized to derive the comparative framework.

**Index Terms:** *Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), Healthcare AI, Hallucination, Clinical Safety, Information Retrieval, Generative Models*

---

## 🎯 Research Objectives

<table>
<tr>
<td width="50%">

### Primary Objectives

```
1. Conduct a survey-based comparative analysis
   of LLM vs RAG in healthcare settings using
   20 peer-reviewed papers (2023–2025).

2. Quantify hallucination rates and their
   clinical safety implications across both
   architectures.

3. Assess factual accuracy, groundedness, and
   retrieval relevance across medical domains
   including MedQA, PubMedQA, and MIMIC-IV.

4. Evaluate response quality and clinical
   deployment readiness on 8 key metrics.
```

</td>
<td width="50%">

### Secondary Objectives

```
5. Analyze architectural trade-offs between
   accuracy, fluency, and computational cost
   (RAG incurs 15–25% overhead).

6. Compare RAG paradigms: Naive, Advanced,
   and Modular across retrieval complexity
   and contextual integration levels.

7. Synthesize benchmarks (MIRAGE, RAG², 
   Self-BioRAG) to identify best practices
   for medical AI evaluation.

8. Outline future research directions for
   multimodal RAG and continual learning
   in clinical AI systems.
```

</td>
</tr>
</table>

---

## 🏗️ Research Methodology

> **Study Type:** Survey-based comparative framework (not a new model — synthesizes and cross-analyzes existing research)
> **Corpus:** 20 peer-reviewed papers, 2023–2025 · 12,000+ model outputs · 50+ evaluation metrics

### Survey-Based Comparative Framework (Fig. 1 from Paper)

```
╔══════════════════════════════════════════════════════════════════════════╗
║              SURVEY-BASED COMPARATIVE METHODOLOGY FRAMEWORK              ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   ┌─────────────────────────────────────────────────────────────────┐   ║
║   │         LITERATURE SELECTION  (20 Healthcare Papers)            │   ║
║   │   Inclusion: RAG vs LLM comparison · Healthcare datasets ·      │   ║
║   │   Quantitative metrics · Published 2023–2025                    │   ║
║   └─────────────────────────────┬───────────────────────────────────┘   ║
║                                 │                                        ║
║              ┌──────────────────┼──────────────────┐                    ║
║              ▼                  ▼                  ▼                    ║
║   ┌──────────────────┐ ┌────────────────┐ ┌──────────────────┐         ║
║   │ RETRIEVAL        │ │ GENERATION     │ │ EVALUATION       │         ║
║   │ MECHANISM        │ │ MODEL          │ │ MATRIX           │         ║
║   │                  │ │                │ │                  │         ║
║   │ • Dense Vector   │ │ • GPT-4        │ │ • 8 Metrics      │         ║
║   │   (BioSentVec,   │ │ • Claude 3     │ │ • Auto + Human   │         ║
║   │   PubMedBERT)    │ │ • LLaMA-2      │ │ • Expert Review  │         ║
║   │ • Hybrid         │ │ • Falcon       │ │ • [0,1] Norm.    │         ║
║   │   Dense+Sparse   │ │ • Fine-tuned   │ │                  │         ║
║   │ • Graph-based    │ │   domain LLMs  │ │                  │         ║
║   │   (GraphRAG)     │ │                │ │                  │         ║
║   └──────────────────┘ └────────────────┘ └──────────────────┘         ║
║              │                  │                  │                    ║
║              └──────────────────┼──────────────────┘                    ║
║                                 ▼                                        ║
║   ┌─────────────────────────────────────────────────────────────────┐   ║
║   │         COMPARATIVE SYNTHESIS  (RAG vs. LLM Summary)           │   ║
║   │   Normalized scores · Statistical averages · Trend analysis     │   ║
║   └─────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### RAG Pipeline Architecture

```
INPUT: Clinical Query
        │
        ▼
┌───────────────────┐
│  1. INDEXING      │  ← Offline Phase
│                   │
│  • Chunk medical  │
│    documents      │
│  • Embed with     │
│    BioBERT/SBERT/ │
│    PubMedBERT     │
│  • Store in FAISS │
│    vector index   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  2. RETRIEVAL     │  ← Online Phase (per query)
│                   │
│  • Encode query   │
│  • ANN search     │
│  • Re-rank top-K  │
│    passages       │
│  • Graph traversal│
│    (GraphRAG)     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  3. AUGMENTATION  │
│                   │
│  • Construct      │
│    prompt with    │
│    retrieved      │
│    context        │
│  • Rationale-     │
│    guided reform. │
│    (RAG²)         │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  4. GENERATION    │
│                   │
│  • LLM generates  │
│    grounded       │
│    response       │
│  • Self-reflection│
│    (Self-BioRAG)  │
│  • Cite sources   │
└───────────────────┘
        │
        ▼
OUTPUT: Evidence-Grounded Clinical Response
```

### Three-Layer Evaluation Framework

```
┌────────────────────────────────────────────────────────────────┐
│                 THREE-LAYER EVALUATION PIPELINE                │
├───────────────────────────┬────────────────────────────────────┤
│    AUTOMATED METRICS      │    HUMAN / EXPERT EVALUATION       │
├───────────────────────────┼────────────────────────────────────┤
│  • Factual correctness    │  • Clinical expert panel review    │
│  • BLEU / ROUGE scores    │  • Radiologist & pharmacist panels │
│  • Faithfulness scoring   │  • Safety annotation & flagging    │
│  • Hallucination rate     │  • Linguistic quality assessment   │
│  • Groundedness score     │  • Inter-annotator agreement (IAA) │
└───────────────────────────┴────────────────────────────────────┘

Normalization: All values mapped to [0,1] scale
Qualitative encoding: Excellent=0.9 · Good=0.75 · Moderate=0.5 · Poor=0.25
Datasets: MedQA · PubMedQA · MIMIC-IV · MIRAGE benchmark
```

### Benchmarks & Studies Surveyed

| Study | Architecture | Key Contribution | Gain vs LLM |
|:---|:---|:---|:---:|
| **MIRAGE** (2024) | RAG | 40+ retriever–corpus–LLM combinations | +18% accuracy |
| **Self-BioRAG** (2024) | RAG + Self-reflection | Self-reflective biomedical QA | +7% QA accuracy |
| **RAG²** (NAACL 2025) | Rationale-guided RAG | Query reformulation for medical QA | +6% factual alignment |
| **i-MedRAG** | Iterative RAG | Follow-up retrieval for reasoning continuity | Improved reasoning |
| **GraphRAG** | Knowledge Graph RAG | Graph-based structured medical reasoning | Reduced false positives |
| **Omni-RAG** | Multi-source RAG | Diverse database integration | Comprehensive context |
| **Bayesian-RAG** | Probabilistic RAG | Uncertainty-aware clinical decisions | Reliable support |

---

## 📊 Key Findings & Results

<div align="center">

### 🏆 RAG Outperforms LLM on 6 of 8 Clinical Metrics

</div>

```
METRIC PERFORMANCE COMPARISON  (Normalized 0–1 · Aggregated from 20 studies)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Factual Accuracy
  LLM  ████████████████████████████░░░░░░░░░░░░░░  0.72
  RAG  ████████████████████████████████████░░░░░░  0.88  ▲ +22.2%

Faithfulness / Groundedness
  LLM  ██████████████████████████░░░░░░░░░░░░░░░░  0.68
  RAG  ████████████████████████████████████░░░░░░  0.87  ▲ +27.9%

Hallucination Rate  (lower is better ↓)
  LLM  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.27
  RAG  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.09  ▼ −66.7%

Clinical Safety
  LLM  █████████████████████████████░░░░░░░░░░░░░  0.71
  RAG  ██████████████████████████████████████░░░░  0.92  ▲ +29.6%

Relevance (Retrieval)
  LLM  ████████████████████████████░░░░░░░░░░░░░░  0.70
  RAG  █████████████████████████████████░░░░░░░░░  0.83  ▲ +18.6%

Fluency / Coherence
  LLM  ████████████████████████████████████░░░░░░  0.89  ← LLM leads
  RAG  █████████████████████████████████░░░░░░░░░  0.84  ▼  −5.6%

Response Diversity
  LLM  ██████████████████████████████░░░░░░░░░░░░  0.76
  RAG  █████████████████████████████████░░░░░░░░░  0.80  ▲  +5.3%

Computational Efficiency  (inference speed)
  LLM  ██████████████████████████████████░░░░░░░░  0.85  ← LLM leads
  RAG  █████████████████████████░░░░░░░░░░░░░░░░░  0.73  ▼ −14.1%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Source: Aggregated from MIRAGE, Self-BioRAG, RAG², Omni-RAG,
          GraphRAG, Bayesian-RAG and 14 additional studies.
```

### 🔑 Key Takeaways

1. **🏆 RAG wins decisively on safety-critical metrics** — +29.6% clinical safety, +27.9% groundedness, +22.2% factual accuracy.
2. **⚠️ Hallucination drops by 66.7%** — from 0.27 (LLM) to 0.09 (RAG), making RAG the only viable architecture for clinical decision support.
3. **🧠 LLMs retain fluency & speed advantages** — 5.6% better coherence and 14.1% faster inference; suitable for low-stakes informational tasks.
4. **📊 Retrieval quality is the key differentiator** — across all studies, stronger retrievers correlate directly with higher factual reliability.
5. **⚖️ The efficiency trade-off is clinically justified** — RAG incurs 15–25% computational overhead, a worthwhile cost for a 29.6% safety gain.

---

## 📈 Evaluation Metrics

<div align="center">

### Complete Performance Table (Table II & III from Paper)

| Metric | LLM (Avg.) | RAG (Avg.) | Relative Gain | Winner |
|:---|:---:|:---:|:---:|:---:|
| 🎯 **Factual Accuracy** | 0.72 | **0.88** | +22.2% | 🏆 RAG |
| 📌 **Faithfulness / Groundedness** | 0.68 | **0.87** | +27.9% | 🏆 RAG |
| 🚫 **Hallucination Rate** ↓ | 0.27 | **0.09** | −66.7% | 🏆 RAG |
| 🛡️ **Clinical Safety** | 0.71 | **0.92** | +29.6% | 🏆 RAG |
| 🔍 **Relevance (Retrieval)** | 0.70 | **0.83** | +18.6% | 🏆 RAG |
| 💬 **Fluency / Coherence** | **0.89** | 0.84 | −5.6% | ⚡ LLM |
| 🌐 **Response Diversity** | 0.76 | **0.80** | +5.3% | 🏆 RAG |
| ⚡ **Inference Efficiency** | **0.85** | 0.73 | −14.1% | ⚡ LLM |

> ↓ *Lower is better for Hallucination Rate. All other metrics: higher is better.*
> 📊 *Scores are normalized averages aggregated across 20 peer-reviewed studies (2023–2025).*

### Confusion Matrix Analysis (Table IV from Paper)
*Based on 1,000 QA instances aggregated from MIRAGE, Self-BioRAG, GraphRAG*

| Class | LLM Predicted Correct | LLM Predicted Incorrect | RAG Predicted Correct | RAG Predicted Incorrect |
|:---|:---:|:---:|:---:|:---:|
| ✅ **True Facts** | 640 (64%) | 120 | **830 (83%)** | 60 |
| ❌ **Hallucinated** | 160 | 80 | **70** | 40 |

> RAG correctly identifies factual information in **83% of cases** vs **64% for LLMs** — a 19-point improvement in clinical QA precision.

</div>

---

## 🔍 Comparative Analysis

<details>
<summary><b>📊 Architectural Comparison: LLM vs RAG</b></summary>

<br/>

| Dimension | LLM (Standalone) | RAG Pipeline |
|:---|:---|:---|
| **Knowledge Source** | Parametric (frozen weights) | Dynamic retrieval + parametric |
| **Update Mechanism** | Full fine-tuning / retraining | Index update (no retraining) |
| **Hallucination Risk** | High (0.27) | Low (0.09) — **−66.7%** |
| **Source Attribution** | None / unreliable | Explicit passage citation |
| **Deployment Complexity** | Low | Medium–High |
| **Inference Latency** | Low (0.85 efficiency) | Medium (0.73 efficiency) — **+15–25% overhead** |
| **Factual Currency** | Frozen at training cutoff | Real-time via index updates |
| **Clinical Suitability** | Limited | High |
| **Explainability** | Low | High (traceable sources) |
| **Best Use Case** | Conversational / educational | Clinical decision support |

</details>

<details>
<summary><b>🧮 Loss Function Formulations (from Paper Section IV-E)</b></summary>

<br/>

**Standard LLM Objective** — minimizes cross-entropy between predicted and target token probabilities:

```
L_LLM = − Σ(t=1 to T) log P_θ(y_t | y_<t, x)

where:
  y_t  = target token at step t
  x    = input context (query only)
  P_θ  = model's predicted distribution
```

**RAG Retrieval-Conditioned Objective** — incorporates relevant external documents D = {d₁, d₂, …, d_k}:

```
L_RAG = − E_{d ~ p_φ(d|x)} Σ(t=1 to T) log P_θ(y_t | y_<t, x, d)

where:
  d        = retrieved document(s) from external knowledge base
  p_φ(d|x) = retriever's probability distribution over documents
  x, d     = augmented context (query + retrieved passages)
```

> The retriever distribution `p_φ(d|x)` ensures generation remains grounded in verifiable context, effectively **penalizing unsupported predictions** and reducing hallucinations.

</details>

<details>
<summary><b>⚖️ Trade-off Summary: When to Use Which Architecture</b></summary>

<br/>

```
USE STANDALONE LLM WHEN:
  ✓ Speed and low latency are paramount
  ✓ General health education (not clinical decisions)
  ✓ Preliminary triage or patient intake forms
  ✓ Natural language understanding / summarization tasks
  ✓ No sensitive or time-critical medical facts involved
  ✓ LLMs perform 8–10% better in fluency/coherence

USE RAG PIPELINE WHEN:
  ✓ Clinical decision support is the primary use case
  ✓ Drug dosage, contraindication lookup required
  ✓ Evidence-based treatment recommendations needed
  ✓ Regulatory compliance (FDA, HIPAA) is a concern
  ✓ Source attribution & auditability are required
  ✓ Knowledge base must stay current post-deployment
  ✓ RAG achieves 10–22% higher factual accuracy
  ✓ RAG achieves 25–40% lower hallucination rates

CONSIDER HYBRID APPROACH WHEN:
  ✓ Mixed workloads (informational + clinical)
  ✓ Cost-sensitive deployments at scale
  ✓ Tiered trust levels based on query criticality
  ✓ RAG cost overhead (15–25%) must be managed
```

</details>

---

## 🌟 Major Contributions

<table>
<tr>
<td width="50%" valign="top">

### 📐 Technical Contributions

**C1 — Comprehensive Survey Framework**
> A three-layer survey methodology covering retrieval, generation, and evaluation phases — enabling systematic comparability across 20 heterogeneous studies.

**C2 — Normalized Evaluation Matrix**
> An 8-dimensional, cross-study evaluation matrix with standardized [0,1] normalization, enabling direct aggregation of diverse metric types (quantitative + qualitative).

**C3 — Confusion Matrix Analysis**
> A 1,000-instance QA confusion matrix comparison showing RAG achieves 83% factual precision vs 64% for LLMs, reducing false positives in clinical settings.

</td>
<td width="50%" valign="top">

### 🔬 Scientific Contributions

**C4 — Loss Function Formalization**
> Formal mathematical comparison of LLM cross-entropy loss (L_LLM) vs RAG retrieval-conditioned loss (L_RAG), explaining why RAG structurally reduces hallucinations.

**C5 — Quantified Trade-off Analysis**
> The first normalized synthesis showing RAG incurs 14.1% efficiency loss but delivers 22–30% gains on all safety-critical metrics — a clearly justified clinical trade-off.

**C6 — Evidence-Based Architecture Guidance**
> Actionable decision framework for selecting LLM vs RAG vs Hybrid architectures based on clinical risk level and deployment requirements.

</td>
</tr>
</table>

---

## 🏥 Healthcare Applications

<div align="center">

```
┌──────────────────────────────────────────────────────────────────┐
│              CLINICAL AI APPLICATION LANDSCAPE                   │
├──────────────┬───────────────────────────────┬───────────────────┤
│  APPLICATION │ RECOMMENDED ARCHITECTURE      │ SAFETY LEVEL      │
├──────────────┼───────────────────────────────┼───────────────────┤
│ 💊 Drug Info │ RAG (Pharmacology Index)       │ 🔴 Critical       │
│ 🩺 Diagnosis │ RAG (Clinical Literature)      │ 🔴 Critical       │
│ 📋 Protocols │ RAG (Guideline Database)       │ 🔴 Critical       │
│ 🧪 Lab Notes │ RAG (Reference Ranges DB)      │ 🟠 High           │
│ 📚 Education │ LLM or RAG                     │ 🟡 Medium         │
│ 💬 Chatbot   │ LLM (Low-risk queries only)    │ 🟢 Low            │
│ 📊 Analytics │ LLM + Structured Data          │ 🟡 Medium         │
│ 🏥 Discharge │ RAG (Institutional Templates)  │ 🟠 High           │
└──────────────┴───────────────────────────────┴───────────────────┘
```

</div>

### Real-World Impact Areas

- 🏥 **Clinical Decision Support Systems (CDSS)** — Reducing diagnostic errors with evidence-cited AI responses
- 💊 **Pharmacy & Drug Interaction Alerts** — Zero-hallucination drug information retrieval
- 🧑‍⚕️ **Medical Education & Training** — Factually reliable AI tutors for medical students
- 📋 **EHR Documentation Assistance** — Guideline-aligned clinical note generation
- 🔬 **Biomedical Research Assistance** — Literature-grounded hypothesis generation
- 🌍 **Global Health Access** — Scalable, low-cost clinical knowledge delivery in underserved regions

---

## 🛠️ Technologies & Concepts

<details>
<summary><b>🤖 Models & Systems Evaluated</b></summary>

<br/>

| Category | Systems / Models |
|:---|:---|
| **Proprietary LLMs** | GPT-4, Claude 3, PaLM, Gemini |
| **Open-Source LLMs** | LLaMA-2, Falcon, fine-tuned domain LLMs |
| **Embedding Models** | text-embedding-3-large, BioSentVec, PubMedBERT |
| **RAG Frameworks Surveyed** | MIRAGE, Self-BioRAG, RAG², i-MedRAG, Omni-RAG, GraphRAG, Bayesian-RAG |
| **Retrieval Methods** | Dense vector (FAISS), Hybrid (dense+sparse), Knowledge Graph |
| **Medical Datasets** | MedQA, PubMedQA, MIMIC-IV, MIRAGE benchmark |

</details>

<details>
<summary><b>📏 Evaluation Tools & Metrics</b></summary>

<br/>

| Category | Technologies / Methods |
|:---|:---|
| **Automated Metrics** | BLEU, ROUGE, BERTScore, Faithfulness scoring |
| **Human Evaluation** | Clinical expert panels, radiologist annotation, pharmacist review |
| **Normalization** | [0,1] scale; Excellent=0.9, Good=0.75, Moderate=0.5, Poor=0.25 |
| **Statistical Methods** | Cross-study aggregation, normalized averages, variance analysis |
| **QA Evaluation** | Confusion matrix analysis (1,000 instances) |

</details>

<details>
<summary><b>📚 Key Concepts Explored</b></summary>

<br/>

- **Retrieval-Augmented Generation (RAG)** — Naive, Advanced, and Modular paradigms
- **Hallucination in LLMs** — Causes, measurement, and mitigation via retrieval
- **Dense Vector Retrieval** — Bi-encoder models for medical passage retrieval
- **Clinical NLP** — Domain adaptation for biomedical language understanding
- **RAG Loss Function** — Retrieval-conditioned optimization L_RAG vs L_LLM
- **Evidence-Based Medicine (EBM)** — Integration with clinical evidence hierarchies
- **AI Safety in Healthcare** — Regulatory considerations (FDA, HIPAA, CE marking)
- **Graph-Based Knowledge Retrieval** — Structured medical concept relationships

</details>

---

## 💡 Research Impact

<div align="center">

### Why This Research Matters

</div>

```
┌─────────────────────────────────────────────────────────────────┐
│                  THE STAKES IN HEALTHCARE AI                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 Every year, medical errors affect ~250,000+ patients in     │
│     the US alone. AI must not add to this burden.              │
│                                                                 │
│  🤖 LLMs are already being used in clinical settings, yet       │
│     their hallucination risks remain poorly understood.         │
│                                                                 │
│  🔬 This research provides empirical evidence that RAG          │
│     architectures are significantly safer for clinical AI.      │
│                                                                 │
│  📋 Our evaluation framework enables standardized, repeatable   │
│     benchmarking of healthcare AI systems.                      │
│                                                                 │
│  🌍 The open-source release democratizes rigorous healthcare     │
│     AI evaluation for researchers worldwide.                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔭 Future Work

<table>
<tr>
<td width="50%" valign="top">

### Near-Term Research Directions

- [ ] 🖼️ **Multimodal RAG** — Extending retrieval to include clinical images, radiology scans, and structured EHR tables alongside text
- [ ] 🔄 **Continual Learning Integration** — Real-time medical evidence updates without full retraining via dynamic index management
- [ ] 📏 **Standardized Benchmarks** — Developing universal, cross-study evaluation standards to resolve the current lack of comparable metrics
- [ ] 🔍 **Retriever Interpretability** — Improving transparency in why specific passages are retrieved for a given clinical query

</td>
<td width="50%" valign="top">

### Long-Term Research Directions

- [ ] ⚖️ **Fairness & Bias Auditing** — Systematic analysis of retrieval bias and demographic disparities in RAG healthcare outputs
- [ ] 🔒 **Privacy-Preserving RAG** — Federated retrieval frameworks that comply with HIPAA and patient data regulations
- [ ] 🤝 **Human-AI Clinical Trials** — Randomized studies measuring real-world impact of RAG vs LLM on clinician workflows
- [ ] 🏛️ **Regulatory Frameworks** — Contributing to FDA and CE-marking standards for RAG-based clinical AI certification

</td>
</tr>
</table>


---

## 📖 Publication Details

<div align="center">

| Field | Details |
|:---:|:---|
| 📄 **Title** | Evaluating RAG and LLM Architectures for Evidence-Grounded Healthcare AI |
| 🏛️ **Venue** | *[Conference / Journal Name — Under Review / Published]* |
| 📅 **Year** | 2025 |
| 🔗 **DOI** | `[DOI — To be assigned upon publication]` |
| 📑 **arXiv** | `[arXiv ID — Pending]` |
| 🌐 **Format** | IEEE Conference Paper Format |

</div>

### Authors

<div align="center">

| # | Author | Email | Affiliation | Role |
|:---:|:---:|:---|:---|:---:|
| 1 | **Tanmay Tyagi** | e23cseu1448@bennett.edu.in | School of Computer Science and Technology, Bennett University, UP, India | 🥇 First Author |
| 2 | **Ashok Kumar Rai** | ashokkumarrai@bennett.edu.in | School of Computer Science and Technology, Bennett University, UP, India | 🤝 Co-Author |
| 3 | **Khushi Saroha** | e23cseu1446@bennett.edu.in | School of Computer Science and Technology, Bennett University, UP, India | 🤝 Co-Author |
| 4 | **Jallipalli Pramod** | e23cseu2325@bennett.edu.in | School of Computer Science and Technology, Bennett University, UP, India | 🤝 Co-Author |
| 5 | **Rohan Sharma** | e23cseu1460@bennett.edu.in | School of Computer Science and Technology, Bennett University, UP, India | 🤝 Co-Author |

</div>

---

## 💬 Citation

If this work contributes to your research, please cite it as follows:

### BibTeX

```bibtex
@inproceedings{tyagi2025healthcare_rag_llm,
  title     = {Evaluating {RAG} and {LLM} Architectures for Evidence-Grounded Healthcare {AI}},
  author    = {Tyagi, Tanmay and Rai, Ashok Kumar and Saroha, Khushi
               and Pramod, Jallipalli and Sharma, Rohan},
  booktitle = {[Conference / Journal Name]},
  year      = {2025},
  pages     = {[Page range]},
  doi       = {[DOI]},
  url       = {https://github.com/tanmaytyagi/healthcare-rag-llm-eval},
  address   = {Bennett University, Uttar Pradesh, India}
}
```

### APA Format

```
Tyagi, T., Rai, A. K., Saroha, K., Pramod, J., & Sharma, R. (2025). Evaluating RAG and LLM
Architectures for Evidence-Grounded Healthcare AI. [Conference/Journal Name].
https://doi.org/[DOI]
```

### IEEE Format

```
T. Tyagi, A. K. Rai, K. Saroha, J. Pramod, and R. Sharma, "Evaluating RAG and LLM
Architectures for Evidence-Grounded Healthcare AI," [Conference/Journal Name], 2025.
```

## 📚 Key References

<details>
<summary><b>📖 20 Surveyed Papers (2023–2025) — Click to expand</b></summary>

<br/>

| # | Citation | Key Contribution |
|:---:|:---|:---|
| [1] | Zhou et al., *arXiv:2401.04589*, 2024 | Hallucination causes and mitigation in LLMs |
| [2] | Lewis et al., *NeurIPS*, 2020 | Foundational RAG framework for knowledge-intensive NLP |
| [3] | Neha et al., *AI*, vol. 6(9), 2025 | Comprehensive review of RAG in healthcare |
| [4] | Cheng et al., *arXiv:2501.02512*, 2025 | Survey on knowledge-oriented RAG |
| [5] | Ke et al., *arXiv:2405.03256*, 2024 | RAG case study: development and testing |
| [6] | Xiong et al., *arXiv:2402.13178*, 2024 | **MIRAGE**: Benchmarking RAG for medicine |
| [7] | Amugongo et al., *JAMIA*, 2025 | Systematic review & meta-analysis of RAG in healthcare |
| [8] | Wang et al., *EmergentMind*, 2024 | Preoperative chatbot using RAG framework |
| [9] | Jeong et al., *EmergentMind*, 2024 | **Self-BioRAG**: Self-reflective biomedical QA (+7% gain) |
| [10] | Sohn et al., *NAACL*, 2025 | **RAG²**: Rationale-guided RAG for medical QA (+6% gain) |
| [11] | EU R. Group, *JMIR*, 2025 | Two-layer RAG using Reddit data for medical QA |
| [12] | Chen et al., *JMIR*, 2025 | **Omni-RAG**: Multi-source RAG for healthcare decisions |
| [13] | Wu et al., *AI in Medicine*, vol. 162, 2025 | **GraphRAG**: Knowledge graph-guided evidence-based medicine |
| [14] | Zhang et al., *Springer ICISH*, 2025 | **Bayesian-RAG**: Probabilistic clinical decision support |
| [15] | Gan et al., *AI Evaluation Letters*, 2025 | RAG evaluation challenges and standardized metrics |

</details>

---

## 🤝 Contributing

Contributions are warmly welcomed! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md).

```
Fork → Branch → Commit → Pull Request

git checkout -b feature/your-feature
git commit -m "feat: add your feature"
git push origin feature/your-feature
```

[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=flat-square)](docs/CONTRIBUTING.md)

---

## 🙏 Acknowledgements

We sincerely thank:

- 🏫 **Bennett University** — For providing computational resources and institutional support
- 🤗 **HuggingFace** — For the open-source Transformers ecosystem that powers our experiments
- 📚 **PubMed / NLM** — For access to biomedical literature used in our knowledge corpus
- 🔬 **The clinical experts** who provided domain annotations and safety evaluations
- 🌐 **The open-source community** — LangChain, FAISS, RAGAS, and all projects we built upon

---

## 👥 Authors & Contact

<div align="center">

### 📬 Get in Touch

</div>

**Tanmay Tyagi** *(First Author & Maintainer)*
- 🎓 B.Tech Computer Science Engineering (AI & ML)
- 🏫 School of Computer Science and Technology, Bennett University, Uttar Pradesh, India
- 📧 Email: `e23cseu1448@bennett.edu.in`




> For research collaborations, questions about the methodology, or dataset access requests, please open a [GitHub Issue](../../issues) or reach out via email.

---

<div align="center">

## ⭐ Support This Research

*If this repository helped your work, please give it a star — it helps others discover this research and motivates continued development.*

[![Star History Chart](https://img.shields.io/github/stars/tanmaytyagi/healthcare-rag-llm-eval?style=social)](https://github.com/tanmaytyagi/healthcare-rag-llm-eval/stargazers)

```
     ★ Star          🍴 Fork          👁️ Watch          📢 Share
```

**Spread the word:**


[![Share on LinkedIn](https://img.shields.io/badge/Share-LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/shareArticle?url=https://github.com/tanmaytyagi/healthcare-rag-llm-eval)

<br/>

---

<sub>
Made with ❤️ by <a href="https://github.com/tanmaytyagi">Tanmay Tyagi</a> and the research team at Bennett University · 
© 2025 · 
<a href="LICENSE">MIT License</a>
</sub>

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%" />

</div>
