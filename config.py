"""
config.py
=========
Central configuration for the Healthcare RAG vs. LLM research project.

All tunable parameters (model names, chunking strategy, paths, evaluation
thresholds) live here so that other modules import a single source of
truth instead of hard-coding values. This mirrors the survey-based
comparative framework described in the source paper:
"Evaluating RAG and LLM Architectures for Evidence-Grounded Healthcare AI".

Environment variables override defaults where noted, which keeps secrets
(e.g. HF_TOKEN) out of source control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
SAMPLE_PAPERS_DIR: Path = DATA_DIR / "sample_medical_papers"
HEALTHCARE_DATASET_CSV: Path = DATA_DIR / "healthcare_dataset.csv"
VECTORSTORE_DIR: Path = BASE_DIR / "vectorstore"
RESULTS_DIR: Path = BASE_DIR / "results"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
METRICS_CSV: Path = RESULTS_DIR / "metrics.csv"
COMPARISON_REPORT_MD: Path = RESULTS_DIR / "comparison_report.md"
EVALUATION_REPORT_MD: Path = RESULTS_DIR / "evaluation_report.md"
LOG_DIR: Path = BASE_DIR / "logs"

for _dir in (DATA_DIR, SAMPLE_PAPERS_DIR, VECTORSTORE_DIR, RESULTS_DIR, PLOTS_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the generative (LLM) backbone shared by both
    the baseline LLM system and the RAG generation stage."""

    # HuggingFace repo id for Llama 3 8B Instruct. Override with env var
    # LLM_MODEL_ID if you are using a gated/local checkpoint or a hosted
    # inference endpoint alias.
    model_id: str = os.getenv("LLM_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")

    # Generation parameters
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))
    repetition_penalty: float = float(os.getenv("LLM_REPETITION_PENALTY", "1.1"))

    # Inference backend: "local" (transformers pipeline), "hf_inference"
    # (HuggingFace Inference Endpoint / API), or "ollama" (local Ollama server).
    backend: str = os.getenv("LLM_BACKEND", "ollama")

    # Used when backend == "hf_inference"
    hf_token: str = os.getenv("HF_TOKEN", "")

    # Used when backend == "ollama"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model_name: str = os.getenv("OLLAMA_MODEL_NAME", "llama3:8b-instruct")


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the dense retrieval embedding model."""

    # SentenceTransformers / HuggingFace embedding model used to vectorize
    # chunks for FAISS indexing. PubMedBERT-style models are domain-tuned;
    # MiniLM is a fast general-purpose fallback for CPU-only environments.
    model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    normalize_embeddings: bool = True


@dataclass(frozen=True)
class ChunkingConfig:
    """Text splitting configuration for PDF ingestion."""

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    separators: tuple = field(
        default_factory=lambda: ("\n\n", "\n", ". ", " ", "")
    )


@dataclass(frozen=True)
class RetrievalConfig:
    """FAISS retrieval configuration."""

    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    score_threshold: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.0"))
    index_name: str = "faiss_index"


@dataclass(frozen=True)
class EvaluationConfig:
    """Thresholds and weights used by evaluation.py to compute composite
    scores. Values are informed by the normalized [0,1] scoring scheme
    described in the source survey (Section III-D)."""

    qualitative_score_map: dict = field(
        default_factory=lambda: {
            "excellent": 0.9,
            "good": 0.75,
            "moderate": 0.5,
            "poor": 0.25,
        }
    )
    # Keyword lists used as a lightweight, dependency-free proxy for
    # clinical-safety screening when an LLM-judge is unavailable.
    unsafe_keywords: tuple = field(
        default_factory=lambda: (
            "stop taking your medication",
            "no need to see a doctor",
            "guaranteed cure",
            "ignore your symptoms",
        )
    )
    random_seed: int = 42


MODEL_CONFIG = ModelConfig()
EMBEDDING_CONFIG = EmbeddingConfig()
CHUNKING_CONFIG = ChunkingConfig()
RETRIEVAL_CONFIG = RetrievalConfig()
EVALUATION_CONFIG = EvaluationConfig()


# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE: Path = LOG_DIR / "healthcare_rag.log"
