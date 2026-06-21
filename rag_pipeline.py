"""
rag_pipeline.py
================
Retrieval-Augmented Generation (RAG) system for healthcare question
answering, implementing the pipeline described in Section III-B of the
source paper:

    PDFs -> Chunking -> Embeddings -> FAISS -> Retriever -> Llama 3 -> Answer

Stages:
  1. PDF ingestion       (PyPDFLoader via LangChain, or raw pypdf fallback)
  2. Text chunking        (RecursiveCharacterTextSplitter)
  3. Embedding generation (HuggingFace / SentenceTransformers)
  4. FAISS vector store   (build, persist, load)
  5. Semantic retrieval   (top-k similarity search)
  6. Context injection    (prompts.RAG_USER_TEMPLATE)
  7. Answer generation    (reuses llm_baseline backends for a fair
                           apples-to-apples comparison against the
                           non-retrieval baseline)

Usage:
    from rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.ingest_pdfs(["data/sample_medical_papers/diabetes_guideline.pdf"])
    response = rag.answer("What is the first-line treatment for type 2 diabetes?")
    print(response.answer)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import config
import prompts
from llm_baseline import BaselineLLM, LLMBackendError, _estimate_tokens
from logging_utils import get_logger

logger = get_logger(__name__)


class IngestionError(RuntimeError):
    """Raised when PDF loading or chunking fails."""


class VectorStoreError(RuntimeError):
    """Raised when FAISS index build/load/save operations fail."""


@dataclass
class RetrievedChunk:
    """A single retrieved passage with provenance metadata."""

    text: str
    source: str
    page: Optional[int]
    score: float


@dataclass
class RAGResponse:
    """Container for a single RAG generation result."""

    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    latency_seconds: float = 0.0
    retrieval_latency_seconds: float = 0.0
    generation_latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def source_list(self) -> List[str]:
        return [c.source for c in self.retrieved_chunks]


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline for healthcare QA.

    The pipeline lazily imports `langchain`, `langchain_community`, and
    `sentence_transformers` so that the rest of the codebase (e.g. the
    baseline LLM or evaluation utilities) can run without these heavier
    dependencies installed.
    """

    def __init__(
        self,
        chunking_config: Optional[config.ChunkingConfig] = None,
        embedding_config: Optional[config.EmbeddingConfig] = None,
        retrieval_config: Optional[config.RetrievalConfig] = None,
        generator: Optional[BaselineLLM] = None,
    ) -> None:
        self.chunk_cfg = chunking_config or config.CHUNKING_CONFIG
        self.embed_cfg = embedding_config or config.EMBEDDING_CONFIG
        self.retrieval_cfg = retrieval_config or config.RETRIEVAL_CONFIG
        # Reuse BaselineLLM purely as the generation backend so that the
        # RAG arm and the no-retrieval arm use an identical inference
        # backend/model, isolating "retrieval vs. no retrieval" as the
        # only experimental variable.
        self.generator = generator or BaselineLLM()

        self._embeddings = None
        self._vectorstore = None
        logger.info(
            "Initialized RAGPipeline (chunk_size=%d, overlap=%d, top_k=%d, embed_model=%s)",
            self.chunk_cfg.chunk_size,
            self.chunk_cfg.chunk_overlap,
            self.retrieval_cfg.top_k,
            self.embed_cfg.model_name,
        )

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    @property
    def embeddings(self):
        """Lazily instantiate the HuggingFace embedding model."""
        if self._embeddings is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                except ImportError as exc:
                    raise VectorStoreError(
                        "Embedding support requires `pip install langchain-huggingface "
                        "sentence-transformers`."
                    ) from exc
            logger.info("Loading embedding model: %s", self.embed_cfg.model_name)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embed_cfg.model_name,
                model_kwargs={"device": self.embed_cfg.device},
                encode_kwargs={"normalize_embeddings": self.embed_cfg.normalize_embeddings},
            )
        return self._embeddings

    # ------------------------------------------------------------------ #
    # Stage 1-2: PDF ingestion + chunking
    # ------------------------------------------------------------------ #
    def load_and_chunk_pdfs(self, pdf_paths: List[str]):
        """Load PDFs and split them into overlapping text chunks.

        Args:
            pdf_paths: Paths to PDF files on disk.

        Returns:
            List of LangChain Document objects (chunked).

        Raises:
            IngestionError: If a PDF cannot be read or no text is extracted.
        """
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise IngestionError(
                "PDF ingestion requires `pip install langchain-community pypdf "
                "langchain-text-splitters`."
            ) from exc

        documents = []
        for pdf_path in pdf_paths:
            path = Path(pdf_path)
            if not path.exists():
                raise IngestionError(f"PDF not found: {pdf_path}")
            try:
                loader = PyPDFLoader(str(path))
                docs = loader.load()
            except Exception as exc:  # noqa: BLE001
                raise IngestionError(f"Failed to load PDF {pdf_path}: {exc}") from exc

            if not docs:
                logger.warning("No content extracted from %s", pdf_path)
                continue

            for d in docs:
                d.metadata["source"] = path.name
            documents.extend(docs)
            logger.info("Loaded %d page(s) from %s", len(docs), path.name)

        if not documents:
            raise IngestionError("No text could be extracted from the provided PDFs.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_cfg.chunk_size,
            chunk_overlap=self.chunk_cfg.chunk_overlap,
            separators=list(self.chunk_cfg.separators),
        )
        chunks = splitter.split_documents(documents)
        logger.info("Split %d page(s) into %d chunk(s)", len(documents), len(chunks))
        return chunks

    # ------------------------------------------------------------------ #
    # Stage 3-4: Embedding + FAISS index build / persist / load
    # ------------------------------------------------------------------ #
    def build_vectorstore(self, chunks) -> None:
        """Embed chunks and build an in-memory FAISS index.

        Args:
            chunks: LangChain Document objects, typically the output of
                load_and_chunk_pdfs().
        """
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise VectorStoreError(
                "FAISS support requires `pip install faiss-cpu langchain-community`."
            ) from exc

        if not chunks:
            raise VectorStoreError("Cannot build a vector store from zero chunks.")

        logger.info("Embedding %d chunks and building FAISS index...", len(chunks))
        self._vectorstore = FAISS.from_documents(chunks, self.embeddings)
        logger.info("FAISS index built with %d vectors.", len(chunks))

    def persist_vectorstore(self, directory: Optional[str] = None) -> None:
        """Save the FAISS index to disk for reuse across sessions."""
        if self._vectorstore is None:
            raise VectorStoreError("No vector store to persist; call build_vectorstore first.")
        target = Path(directory) if directory else config.VECTORSTORE_DIR
        target.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(target), index_name=self.retrieval_cfg.index_name)
        logger.info("Persisted FAISS index to %s", target)

    def load_vectorstore(self, directory: Optional[str] = None) -> None:
        """Load a previously persisted FAISS index from disk."""
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise VectorStoreError(
                "FAISS support requires `pip install faiss-cpu langchain-community`."
            ) from exc

        source = Path(directory) if directory else config.VECTORSTORE_DIR
        index_file = source / f"{self.retrieval_cfg.index_name}.faiss"
        if not index_file.exists():
            raise VectorStoreError(f"No FAISS index found at {index_file}")

        self._vectorstore = FAISS.load_local(
            str(source),
            self.embeddings,
            index_name=self.retrieval_cfg.index_name,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded FAISS index from %s", source)

    def ingest_pdfs(self, pdf_paths: List[str], persist: bool = True) -> None:
        """Convenience method: load, chunk, embed, index, and (optionally)
        persist PDFs in a single call."""
        chunks = self.load_and_chunk_pdfs(pdf_paths)
        self.build_vectorstore(chunks)
        if persist:
            self.persist_vectorstore()

    # ------------------------------------------------------------------ #
    # Stage 5: Semantic retrieval
    # ------------------------------------------------------------------ #
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Retrieve the top-k most relevant chunks for a question.

        Args:
            question: The user's natural-language query.
            top_k: Override the configured number of chunks to retrieve.

        Returns:
            List of RetrievedChunk, ordered by descending relevance
            (ascending FAISS L2 distance is converted to a similarity-like
            score for readability).
        """
        if self._vectorstore is None:
            raise VectorStoreError(
                "No vector store loaded. Call ingest_pdfs() or load_vectorstore() first."
            )
        k = top_k or self.retrieval_cfg.top_k
        results = self._vectorstore.similarity_search_with_score(question, k=k)

        retrieved = []
        for doc, distance in results:
            # FAISS returns L2 distance for normalized embeddings; convert
            # to a bounded [0,1]-ish similarity score for reporting.
            similarity = 1.0 / (1.0 + float(distance))
            retrieved.append(
                RetrievedChunk(
                    text=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    score=round(similarity, 4),
                )
            )
        logger.info("Retrieved %d chunk(s) for query: %.60s...", len(retrieved), question)
        return retrieved

    # ------------------------------------------------------------------ #
    # Stage 6-7: Context injection + answer generation
    # ------------------------------------------------------------------ #
    def answer(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """Run the full retrieve -> inject -> generate pipeline for a question.

        Args:
            question: The healthcare question posed by the user.
            top_k: Optional override for number of retrieved chunks.

        Returns:
            RAGResponse containing the answer, retrieved sources, latency
            breakdown, and token usage estimates.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        overall_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        retrieved_chunks = self.retrieve(question, top_k=top_k)
        retrieval_latency = time.perf_counter() - retrieval_start

        context_block = prompts.format_context_chunks([c.text for c in retrieved_chunks])
        system_prompt = prompts.RAG_SYSTEM_PROMPT
        user_prompt = prompts.RAG_USER_TEMPLATE.format(
            context=context_block, question=question.strip()
        )

        generation_start = time.perf_counter()
        try:
            answer_text = self.generator._dispatch(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG generation failed")
            raise LLMBackendError(str(exc)) from exc
        generation_latency = time.perf_counter() - generation_start

        overall_latency = time.perf_counter() - overall_start
        prompt_tokens = _estimate_tokens(system_prompt + user_prompt)
        completion_tokens = _estimate_tokens(answer_text)

        logger.info(
            "RAG answered in %.2fs (retrieval=%.2fs, generation=%.2fs, %d completion tokens)",
            overall_latency,
            retrieval_latency,
            generation_latency,
            completion_tokens,
        )

        return RAGResponse(
            question=question,
            answer=answer_text.strip(),
            retrieved_chunks=retrieved_chunks,
            latency_seconds=overall_latency,
            retrieval_latency_seconds=retrieval_latency,
            generation_latency_seconds=generation_latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


if __name__ == "__main__":
    # Quick manual smoke test: `python rag_pipeline.py`
    # Requires at least one PDF in data/sample_medical_papers/.
    sample_dir = config.SAMPLE_PAPERS_DIR
    pdfs = sorted(str(p) for p in sample_dir.glob("*.pdf"))

    if not pdfs:
        print(
            f"No PDFs found in {sample_dir}. Add a medical PDF there and re-run "
            "to test the full ingestion -> retrieval -> generation pipeline."
        )
    else:
        rag = RAGPipeline()
        rag.ingest_pdfs(pdfs)
        sample_question = "What is the recommended first-line treatment discussed in this document?"
        try:
            result = rag.answer(sample_question)
            print(f"Q: {sample_question}\n\nA: {result.answer}\n")
            print(f"Sources: {result.source_list}")
            print(f"Latency: {result.latency_seconds:.2f}s | Tokens: {result.total_tokens}")
        except LLMBackendError as e:
            print(f"[Backend unavailable in this environment] {e}")
