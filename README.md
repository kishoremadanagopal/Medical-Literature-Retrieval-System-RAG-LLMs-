# 🧬 Medical Literature Retrieval System (RAG + LLMs)

End-to-end AI/NLP pipeline for biomedical research. Processes 50K+ PubMed articles and enables natural-language search and Q&A over the corpus using **BioBERT embeddings**, a **Chroma + FAISS hybrid vector store**, **LangChain** orchestration, and **Claude** for generation.

> **Status:** ✅ Complete — 83 tests passing. See [Roadmap](#roadmap).

---

## 🎯 Key Results

| Metric | Value | How it's measured |
|---|---|---|
| Classification accuracy (domain tagging) | **78%** | Held-out test set of 2K manually-labeled articles |
| Query relevance lift vs. BM25 baseline | **+40%** | NDCG@10 on 200-query eval set |
| Retrieval latency (p95) | **30% faster** | FAISS IVF vs. Chroma brute-force, same corpus |
| Corpus size | **50K+ articles** | Broad-scope PubMed ingestion |

*See [`src/evaluation/`](src/evaluation/) for the eval harness that produces these numbers.*

---

## 🏗️ Architecture

```
┌─────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│  PubMed (NCBI)  │──▶│  Ingestion       │──▶│  Raw JSON           │
│  Entrez API     │   │  (Biopython)     │   │  data/raw/          │
└─────────────────┘   └──────────────────┘   └──────────┬──────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  Chunking +        │
                                              │  Cleaning          │
                                              └─────────┬──────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  BioBERT Embedder  │
                                              │  (HuggingFace)     │
                                              └─────────┬──────────┘
                                                        │
                          ┌─────────────────────────────┼─────────────────────────────┐
                          │                                                            │
                ┌─────────▼──────────┐                                       ┌────────▼─────────┐
                │  Chroma            │                                       │  FAISS (IVF)     │
                │  (persistent +     │                                       │  (fast ANN       │
                │  metadata filter)  │                                       │  similarity)     │
                └─────────┬──────────┘                                       └────────┬─────────┘
                          │                                                            │
                          └─────────────────────────────┬──────────────────────────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  Hybrid Retriever  │
                                              │  (LangChain)       │
                                              └─────────┬──────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  Claude (Anthropic)│
                                              │  Answer + Citations│
                                              └─────────┬──────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  Streamlit Demo    │
                                              └────────────────────┘
```

**Why both Chroma and FAISS?**
Chroma gives persistence and rich metadata filtering (author, year, MeSH terms). FAISS gives sub-millisecond nearest-neighbor search at scale. We use Chroma as the source of truth and FAISS as a speed layer — measured in the eval suite.

---

## 📁 Project Structure

```
medical-rag/
├── app.py                 # Streamlit demo UI (Step 7)
├── config/                # Pydantic-validated settings, logging
│   ├── config.py
│   └── logging_config.py
├── src/
│   ├── ingestion/         # Step 2 — PubMed fetcher + XML parser
│   ├── processing/        # Step 3 — chunker + BioBERT embedder
│   ├── retrieval/         # Step 4 — Chroma + FAISS + hybrid retriever
│   ├── generation/        # Step 5 — prompts + RAG chain with Claude
│   └── evaluation/        # Step 6 — metrics + eval harness
├── scripts/               # CLI entry points for each step
├── data/                  # (gitignored) raw, processed, vector stores
├── notebooks/             # Exploration + demos
├── tests/                 # 83 pytest tests
├── docs/                  # Architecture notes, design decisions
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- ~20 GB free disk (for 50K embedded articles + vector indices)
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com/))
- NCBI account email (free, required by ToS)

### Setup

```bash
# 1. Clone and enter
git clone <your-repo-url> medical-rag && cd medical-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY and NCBI_EMAIL

# 5. Verify setup
python scripts/verify_setup.py
```

If `verify_setup.py` prints all green checks, you're ready for the pipeline.

### Full Pipeline

```bash
# Step 2: Ingest 50K PubMed articles (~2-4 hours)
python scripts/run_ingestion.py
python scripts/run_ingestion.py --resume    # if interrupted

# Step 3: Chunk articles + embed with BioBERT (~30-60 min)
python scripts/run_processing.py

# Step 4: Build vector store indices (~5 min)
python scripts/run_indexing.py

# Step 5: Query the system
python scripts/run_query.py "What are CRISPR off-target effects?"
python scripts/run_query.py --interactive

# Step 6: Run evaluation suite
python scripts/run_evaluation.py --output results.json

# Step 7: Launch the demo UI
streamlit run app.py
```

### Quick Test (skip the 4-hour wait)

```bash
# Ingest just 1K articles (~5 min), then run the full pipeline
python scripts/run_ingestion.py --count 1000
python scripts/run_processing.py
python scripts/run_indexing.py
streamlit run app.py
```

---

## 🗺️ Roadmap

- [x] **Step 1** — Project scaffolding, config, logging
- [x] **Step 2** — PubMed ingestion (50K articles via Entrez)
- [x] **Step 3** — Chunking + BioBERT embedding pipeline
- [x] **Step 4** — Chroma (persistent) + FAISS (fast ANN) vector stores
- [x] **Step 5** — LangChain RAG chain with Claude + citation tracking
- [x] **Step 6** — Evaluation harness (retrieval + generation metrics)
- [x] **Step 7** — Streamlit demo UI

---

## 🛠️ Tech Stack

**Orchestration:** LangChain
**Embeddings:** BioBERT (`dmis-lab/biobert-v1.1`) via HuggingFace
**Vector Stores:** Chroma (persistent) + FAISS (ANN)
**LLM:** Claude (Anthropic API)
**Data Source:** PubMed via Biopython/Entrez
**Demo:** Streamlit
**Evaluation:** RAGAs + custom metrics
**Language:** Python 3.10+

---

## 🧪 Testing

```bash
# Run the full test suite (83 tests)
pytest tests/ -v

# Run tests for a specific module
pytest tests/test_parser.py -v       # Step 2: XML parsing
pytest tests/test_chunker.py -v      # Step 3: chunking logic
pytest tests/test_retrieval.py -v    # Step 4: vector stores
pytest tests/test_rag_chain.py -v    # Step 5: RAG chain + prompts
pytest tests/test_evaluation.py -v   # Step 6: metrics
```

All tests run without API keys or data — the Claude API is mocked and vector stores use synthetic data.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

## ✍️ Author

Built as a portfolio project to demonstrate production-grade RAG architecture on a domain-specific corpus.
