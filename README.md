# LangGraph RAG Python Code Assistant

A small research/demo project that builds a Retrieval-Augmented Generation (RAG) assistant for Python code using LangChain-style components, Chroma vector store, and an OpenRouter-compatible LLM. The assistant can generate code and explain code based on the HumanEval dataset as a local knowledge source.

This repository is intended as a compact demo and learning reference — not a production-ready system.

## Features
- Loads the HumanEval dataset and indexes it into Chroma vectors
- Uses sentence-transformers embeddings for retrieval
- Provides two RAG flows: code generation and explanation
- A simple state machine and chat loop for interactive use

## Requirements
- Python 3.9+ (3.10 or 3.11 recommended)
- The packages listed in `requirements.txt` (install with pip)
- An OpenRouter-compatible API key (this project expects it in the code by default — see Security)

## Quick setup

1. Create and activate a virtual environment (recommended):

   - Windows (PowerShell):

     ```powershell
     python -m venv .venv; .\.venv\Scripts\Activate.ps1
     ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. (Optional) If you want to persist vectors to a different directory, edit the call in `rag_langchain.setup_rag_pipeline` or pass a different `persist_directory`.

4. Run the assistant:

   ```powershell
   python main.py
   ```

## Files overview
- `main.py` — entrypoint with chat loop and system initialization
- `rag_langchain.py` — builds the RAG pipeline, loads HumanEval, vectorstore, retriever and the LLM chains
- `nodes_langchain.py` — small node functions used by the graph/state machine (chat, router, generate/explain)
- `state.py` — typed assistant state definition
- `graph.py` — LangGraph graph wiring (state machine). Open this file to inspect how nodes are connected.
- `plot.py` — helper to save a PNG of the LangGraph graph (called by `main.py`)
- `chroma_langchain/` — directory used by Chroma to persist storage (already contains sample db files in this repo)

## Usage examples
- Generate code: "Generate a function to calculate factorial"
- Explain code: "Explain how binary search works"

## Security & API keys
This project currently contains a hard-coded API key placeholder in `rag_langchain.py`. Do NOT commit real API keys to source control. Recommended approaches:

- Use environment variables (e.g., `OPENROUTER_API_KEY`) and read them in `rag_langchain.py`.
- Use a secrets manager or local .env file (with `python-dotenv`) and ensure `.env` is in `.gitignore`.

Before using this repo for anything other than experimentation, move any keys out of source files.

## Troubleshooting
- If HumanEval dataset download fails, ensure `datasets` package is installed and you have network access. You can also provide your own documents and bypass `load_humaneval_documents`.
- If Chroma errors on startup, delete or move `chroma_langchain/chroma.sqlite3` and let the vector store rebuild.
- If embeddings fail to load, ensure `sentence-transformers` and the `sentence-transformers/all-MiniLM-L6-v2` model are available; install packages listed in `requirements.txt`.

## Development notes and caveats
- This is a demo project. It assumes local compute and small datasets. For production, consider managed vector stores, secure key management, proper retry/timeout handling for LLM calls, and privacy controls.
- The RAG chains in `rag_langchain.py` use a custom prompt and combine a retriever + prompt + LLM pipeline. You can adapt prompts to your needs.
