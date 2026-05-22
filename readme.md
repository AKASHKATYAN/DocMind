# DocMind

> Upload any PDF. Ask anything. Get answers grounded strictly in your document.

![Python](https://img.shields.io/badge/Python-3.10%2B-c9f25d?style=flat-square&labelColor=0d0d0d&color=c9f25d)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-c9f25d?style=flat-square&labelColor=0d0d0d&color=c9f25d)
![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-c9f25d?style=flat-square&labelColor=0d0d0d&color=c9f25d)
![License](https://img.shields.io/badge/License-MIT-c9f25d?style=flat-square&labelColor=0d0d0d&color=c9f25d)

---

## Overview

**DocMind** is a local-first Retrieval-Augmented Generation (RAG) application that lets you upload a PDF document and have a context-aware conversation with it. Every answer is derived strictly from the content of the uploaded file — the model will never hallucinate outside the document.

Built with LangChain, Chroma, HuggingFace embeddings, and Mistral AI, wrapped in a clean dark-mode Streamlit interface.

---

## Features

- **PDF Upload & Processing** — Upload any PDF directly through the browser; no file size tricks or pre-processing needed
- **In-Memory Vector Store** — Each upload creates a fresh Chroma vector store in memory, so sessions never bleed into each other
- **MMR Retrieval** — Maximal Marginal Relevance retrieval fetches the most relevant *and* diverse chunks, reducing redundancy in context
- **Strict Context Grounding** — The LLM is instructed to answer only from the retrieved context; if the answer isn't there, it says so
- **Persistent Chat History** — Conversation accumulates within a session via Streamlit session state
- **Dark Mode UI** — Fully themed dark interface with zero white flash on any element, including the file uploader and chat input

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / App** | [Streamlit](https://streamlit.io) |
| **LLM** | [Mistral AI](https://mistral.ai) via `langchain-mistralai` (`mistral-small-2506`) |
| **Embeddings** | [HuggingFace](https://huggingface.co) — `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | [ChromaDB](https://www.trychroma.com) (in-memory, per session) |
| **Orchestration** | [LangChain](https://python.langchain.com) |
| **PDF Parsing** | `PyPDFLoader` via `langchain-community` |
| **Text Splitting** | `RecursiveCharacterTextSplitter` — 1000 token chunks, 200 overlap |
| **Secrets Management** | Streamlit Secrets (`st.secrets`) |

---

## Project Structure

```
DocMind/
├── app.py                   # Main Streamlit application
├── create_database.py       # Standalone script to pre-build a Chroma DB from a local PDF
├── requirements.txt         # Python dependencies
├── .gitignore
├── README.md
└── .streamlit/
    └── secrets.toml         # Local secrets (never committed)
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- A [Mistral AI API key](https://console.mistral.ai/)
- Git

---

### 1. Clone the repository

```bash
git clone https://github.com/your-username/docmind.git
cd docmind
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv myvenv
myvenv\Scripts\activate

# macOS / Linux
python -m venv myvenv
source myvenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` embedding model (~90MB) from HuggingFace. This is cached locally after the first download.

### 4. Configure your API key

Create the secrets file for local development:

```bash
mkdir .streamlit
```

Create `.streamlit/secrets.toml` and add:

```toml
MISTRAL_API_KEY = "your-mistral-api-key-here"
```

> This file is listed in `.gitignore` and will never be committed.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. **Upload** your PDF using the file uploader in the left sidebar
2. Click **Process Document** — the document is chunked and embedded into a vector store
3. Once the status shows **● Ready**, type your question in the chat input at the bottom
4. DocMind retrieves the most relevant passages and generates a grounded answer

---

## Deployment on Streamlit Cloud

1. Push your repository to GitHub (`.streamlit/secrets.toml` stays local — it's gitignored)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Under **Settings → Secrets**, add:

```toml
MISTRAL_API_KEY = "your-mistral-api-key-here"
```

4. Deploy — no other configuration needed

---

## RAG Configuration

| Parameter | Value | Description |
|---|---|---|
| `chunk_size` | 1000 | Characters per chunk |
| `chunk_overlap` | 200 | Overlap between chunks |
| `search_type` | `mmr` | Maximal Marginal Relevance |
| `k` | 4 | Chunks returned per query |
| `fetch_k` | 25 | Candidate pool for MMR |
| `lambda_mult` | 0.7 | Relevance vs. diversity balance |
| `temperature` | 0.5 | LLM response variability |

---

## Known Limitations

- The vector store is **in-memory only** — uploading a new PDF replaces the previous session entirely
- Very large PDFs (500+ pages) may take longer to process during embedding
- Answers are limited to what the document explicitly states; inferential reasoning across gaps is intentional restricted

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with LangChain · ChromaDB · Mistral AI · Streamlit</sub>
</div>