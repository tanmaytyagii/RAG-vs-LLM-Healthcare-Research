"""
app.py
======
Streamlit interface for the Healthcare RAG vs. LLM research project.

Pages:
  1. Upload Medical PDFs   - ingest PDFs into the FAISS vector store
  2. Ask Questions          - query either system independently
  3. Compare LLM vs RAG     - side-by-side answer comparison
  4. Evaluation Dashboard   - visualize aggregated metrics from results/

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import config
from logging_utils import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Healthcare RAG vs. LLM Research",
    page_icon="🩺",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Cached resource loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_baseline_llm():
    from llm_baseline import BaselineLLM

    return BaselineLLM()


@st.cache_resource(show_spinner=False)
def get_rag_pipeline():
    from rag_pipeline import RAGPipeline

    return RAGPipeline()


def vectorstore_exists() -> bool:
    index_file = config.VECTORSTORE_DIR / f"{config.RETRIEVAL_CONFIG.index_name}.faiss"
    return index_file.exists()


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🩺 Healthcare RAG Research")
st.sidebar.caption(
    "Evaluating RAG and LLM Architectures for Evidence-Grounded Healthcare AI"
)
page = st.sidebar.radio(
    "Navigate",
    [
        "📄 Upload Medical PDFs",
        "❓ Ask Questions",
        "⚖️ Compare LLM vs RAG",
        "📊 Evaluation Dashboard",
    ],
)

st.sidebar.divider()
st.sidebar.markdown(
    f"**Backend:** `{config.MODEL_CONFIG.backend}`  \n"
    f"**Generator model:** `{config.MODEL_CONFIG.ollama_model_name if config.MODEL_CONFIG.backend == 'ollama' else config.MODEL_CONFIG.model_id}`  \n"
    f"**Embedding model:** `{config.EMBEDDING_CONFIG.model_name}`  \n"
    f"**Top-K retrieval:** `{config.RETRIEVAL_CONFIG.top_k}`"
)
st.sidebar.caption(
    "⚠️ This tool is for research purposes only and does not provide medical advice."
)


# --------------------------------------------------------------------------- #
# Page 1: Upload Medical PDFs
# --------------------------------------------------------------------------- #
if page == "📄 Upload Medical PDFs":
    st.title("📄 Upload Medical PDFs")
    st.write(
        "Upload one or more medical/healthcare PDF documents (clinical guidelines, "
        "research papers, drug information sheets) to build the RAG knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF file(s)", type=["pdf"], accept_multiple_files=True
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        persist = st.checkbox("Persist FAISS index to disk", value=True)
    with col2:
        save_copy = st.checkbox(
            "Save a copy to data/sample_medical_papers/", value=True
        )

    if st.button("🔄 Ingest PDFs", type="primary", disabled=not uploaded_files):
        saved_paths = []
        for uf in uploaded_files:
            if save_copy:
                target_path = config.SAMPLE_PAPERS_DIR / uf.name
            else:
                target_path = Path("/tmp") / uf.name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(uf.getbuffer())
            saved_paths.append(str(target_path))

        with st.spinner(f"Ingesting {len(saved_paths)} PDF(s): chunking → embedding → FAISS indexing..."):
            try:
                rag = get_rag_pipeline()
                rag.ingest_pdfs(saved_paths, persist=persist)
                st.success(
                    f"✅ Successfully ingested {len(saved_paths)} PDF(s) into the vector store."
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("PDF ingestion failed")
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    st.subheader("Current Knowledge Base")
    existing_pdfs = sorted(config.SAMPLE_PAPERS_DIR.glob("*.pdf"))
    if existing_pdfs:
        st.write(f"**{len(existing_pdfs)} PDF(s)** found in `data/sample_medical_papers/`:")
        for p in existing_pdfs:
            st.markdown(f"- `{p.name}`")
    else:
        st.info("No PDFs found yet. Upload documents above to get started.")

    st.write(
        f"**Vector store status:** "
        f"{'✅ Index found on disk' if vectorstore_exists() else '⚠️ No persisted index yet'}"
    )


# --------------------------------------------------------------------------- #
# Page 2: Ask Questions
# --------------------------------------------------------------------------- #
elif page == "❓ Ask Questions":
    st.title("❓ Ask a Healthcare Question")

    system_choice = st.radio(
        "Choose system", ["Baseline LLM (no retrieval)", "RAG (retrieval-augmented)"], horizontal=True
    )
    question = st.text_area(
        "Your question",
        placeholder="e.g. What are the early warning signs of sepsis?",
        height=100,
    )

    if st.button("🚀 Get Answer", type="primary", disabled=not question.strip()):
        from llm_baseline import LLMBackendError

        if system_choice.startswith("Baseline"):
            with st.spinner("Generating answer with the baseline LLM..."):
                try:
                    llm = get_baseline_llm()
                    result = llm.answer(question)
                    st.markdown("### Answer")
                    st.write(result.answer)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Latency", f"{result.latency_seconds:.2f}s")
                    c2.metric("Tokens (est.)", result.total_tokens)
                    c3.metric("Retrieval used", "No")
                except LLMBackendError as exc:
                    st.error(f"LLM backend unavailable: {exc}")
        else:
            if not vectorstore_exists():
                st.warning(
                    "No FAISS index found. Please upload and ingest PDFs first on the "
                    "'Upload Medical PDFs' page."
                )
            else:
                with st.spinner("Retrieving context and generating grounded answer..."):
                    try:
                        rag = get_rag_pipeline()
                        if rag._vectorstore is None:
                            rag.load_vectorstore()
                        result = rag.answer(question)
                        st.markdown("### Answer")
                        st.write(result.answer)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Latency", f"{result.latency_seconds:.2f}s")
                        c2.metric("Tokens (est.)", result.total_tokens)
                        c3.metric("Chunks retrieved", len(result.retrieved_chunks))

                        with st.expander("📚 View retrieved sources"):
                            for i, chunk in enumerate(result.retrieved_chunks, start=1):
                                st.markdown(
                                    f"**[Source {i}]** `{chunk.source}` "
                                    f"(page {chunk.page}, score {chunk.score})"
                                )
                                st.caption(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("RAG answer generation failed")
                        st.error(f"RAG pipeline error: {exc}")


# --------------------------------------------------------------------------- #
# Page 3: Compare LLM vs RAG
# --------------------------------------------------------------------------- #
elif page == "⚖️ Compare LLM vs RAG":
    st.title("⚖️ Compare LLM vs RAG")
    st.write("Ask the same question to both systems side-by-side.")

    question = st.text_area(
        "Question to compare",
        placeholder="e.g. What is the first-line treatment for type 2 diabetes?",
        height=100,
    )

    if st.button("⚔️ Run Comparison", type="primary", disabled=not question.strip()):
        from llm_baseline import LLMBackendError

        col_llm, col_rag = st.columns(2)

        with col_llm:
            st.subheader("🤖 Baseline LLM")
            with st.spinner("Generating..."):
                try:
                    llm = get_baseline_llm()
                    llm_result = llm.answer(question)
                    st.write(llm_result.answer)
                    st.caption(
                        f"Latency: {llm_result.latency_seconds:.2f}s | "
                        f"Tokens: {llm_result.total_tokens} | Retrieval: None"
                    )
                except LLMBackendError as exc:
                    st.error(f"Unavailable: {exc}")
                    llm_result = None

        with col_rag:
            st.subheader("📚 RAG System")
            if not vectorstore_exists():
                st.warning("No FAISS index found. Ingest PDFs first.")
                rag_result = None
            else:
                with st.spinner("Retrieving + generating..."):
                    try:
                        rag = get_rag_pipeline()
                        if rag._vectorstore is None:
                            rag.load_vectorstore()
                        rag_result = rag.answer(question)
                        st.write(rag_result.answer)
                        st.caption(
                            f"Latency: {rag_result.latency_seconds:.2f}s | "
                            f"Tokens: {rag_result.total_tokens} | "
                            f"Sources: {', '.join(rag_result.source_list) or 'none'}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("RAG comparison failed")
                        st.error(f"RAG error: {exc}")
                        rag_result = None

        if llm_result and rag_result:
            st.divider()
            st.subheader("📈 Quick Metrics")
            from evaluation import evaluate_llm_response, evaluate_rag_response, save_metrics_csv

            eval_llm = evaluate_llm_response(llm_result)
            eval_rag = evaluate_rag_response(rag_result)

            metric_df = pd.DataFrame(
                [
                    {
                        "System": "LLM",
                        "Faithfulness": eval_llm.faithfulness,
                        "Hallucination Rate": eval_llm.hallucination_rate,
                        "Clinical Safety": eval_llm.clinical_safety_score,
                        "Latency (s)": eval_llm.latency_seconds,
                        "Tokens": eval_llm.token_usage,
                    },
                    {
                        "System": "RAG",
                        "Faithfulness": eval_rag.faithfulness,
                        "Hallucination Rate": eval_rag.hallucination_rate,
                        "Clinical Safety": eval_rag.clinical_safety_score,
                        "Latency (s)": eval_rag.latency_seconds,
                        "Tokens": eval_rag.token_usage,
                    },
                ]
            )
            st.dataframe(metric_df, use_container_width=True, hide_index=True)

            if st.button("💾 Save this comparison to results/metrics.csv"):
                save_metrics_csv([eval_llm, eval_rag])
                st.success("Saved to results/metrics.csv")


# --------------------------------------------------------------------------- #
# Page 4: Evaluation Dashboard
# --------------------------------------------------------------------------- #
elif page == "📊 Evaluation Dashboard":
    st.title("📊 Evaluation Dashboard")

    if not config.METRICS_CSV.exists():
        st.info(
            "No evaluation data yet. Run comparisons on the 'Compare LLM vs RAG' page "
            "(and save them), or run `python evaluation.py` / the batch experiment "
            "notebook to populate `results/metrics.csv`."
        )
    else:
        df = pd.read_csv(config.METRICS_CSV)
        st.write(f"Loaded **{len(df)}** evaluation records from `results/metrics.csv`.")

        from evaluation import aggregate_by_system

        agg = aggregate_by_system(df)
        st.subheader("Aggregated Averages by System")
        st.dataframe(agg, use_container_width=True)

        st.subheader("Visual Comparisons")
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Accuracy", "Hallucination", "Latency", "Clinical Safety"]
        )

        with tab1:
            if "factual_accuracy" in agg.columns:
                st.bar_chart(agg["factual_accuracy"])
        with tab2:
            if "hallucination_rate" in agg.columns:
                st.bar_chart(agg["hallucination_rate"])
        with tab3:
            if "latency_seconds" in agg.columns:
                st.bar_chart(agg["latency_seconds"])
        with tab4:
            if "clinical_safety_score" in agg.columns:
                st.bar_chart(agg["clinical_safety_score"])

        st.divider()
        st.subheader("Raw Records")
        st.dataframe(df, use_container_width=True)

        st.subheader("📥 Export")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Generate comparison_report.md"):
                from evaluation import generate_comparison_report

                path = generate_comparison_report(df)
                st.success(f"Saved to {path}")
        with c2:
            if st.button("Generate evaluation_report.md"):
                from evaluation import generate_evaluation_report

                path = generate_evaluation_report(df)
                st.success(f"Saved to {path}")
