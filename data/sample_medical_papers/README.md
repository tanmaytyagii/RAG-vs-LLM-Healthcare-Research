# Sample Medical Papers

Place the PDF documents you want the RAG system to retrieve from in this
directory (clinical guidelines, medical reference papers, drug
prescribing information, etc.).

No PDFs are bundled with this repository because clinical guideline
documents are typically copyrighted or institution-specific. To get
started quickly, you can populate this folder with publicly available,
redistributable PDFs such as:

- WHO clinical guideline summaries (https://www.who.int/publications)
- CDC clinical guidance documents (https://www.cdc.gov)
- Open-access papers from PubMed Central (https://www.ncbi.nlm.nih.gov/pmc/)
- NIH StatPearls / MedlinePlus exportable summaries

## Naming convention

The sample `data/healthcare_dataset.csv` references the following
filenames in its `relevant_sources` column (used to compute retrieval
precision/recall in `evaluation.py`). For the precision/recall metric to
be meaningful out of the box, name your PDFs to match:

```
sepsis_guideline.pdf
diabetes_management_guideline.pdf
pneumonia_clinical_guideline.pdf
hypertension_guideline.pdf
ace_inhibitor_prescribing_information.pdf
colorectal_cancer_screening_guideline.pdf
dsm5_depression_criteria.pdf
mi_management_guideline.pdf
vte_prevention_guideline.pdf
dka_management_protocol.pdf
```

If you use your own documents and questions instead, simply update
`data/healthcare_dataset.csv` to match your filenames — the pipeline
does not require these exact names, this is only to align with the
bundled example dataset.

## Ingesting your PDFs

Once PDFs are in this folder, ingest them via:

```bash
python -c "
from rag_pipeline import RAGPipeline
import config, glob

rag = RAGPipeline()
pdfs = glob.glob(str(config.SAMPLE_PAPERS_DIR / '*.pdf'))
rag.ingest_pdfs(pdfs)
"
```

This builds and persists a FAISS index to `vectorstore/`, which the
Streamlit app (`app.py`) and `notebooks/experiments.ipynb` will load
automatically.
