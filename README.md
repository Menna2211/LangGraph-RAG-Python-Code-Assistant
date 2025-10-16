# LangGraph RAG Python Code Assistant

A lightweight Retrieval-Augmented Generation (RAG) prototype that wires a LangChain-style retrieval pipeline to a small state/graph-driven assistant. Provides a FastAPI HTTP API for chat-style queries that combine local context retrieval (Chroma DB) with a language model backend.

## Highlights
- FastAPI server exposing a simple chat API
- RAG pipeline setup in `rag_langchain.py`
- Simple session state management in `app.py` and `state.py`
- Local Chroma database stored under `chroma_langchain/`

## Requirements
- Python 3.10+
- A Python virtual environment (recommended)
- Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1  # PowerShell on Windows
pip install -r requirements.txt
```

Note: The project may depend on external LLM/embedding providers (OpenAI, Azure, etc.). Configure any required API keys or provider settings as environment variables. Check `rag_langchain.py` for provider-specific configuration and required environment variables.

## Quick start (run locally)

Start the API with Uvicorn:

```powershell
# from repository root
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/ to see the API info.

## API

Endpoints (from `app.py`):

- GET / : Basic API info and endpoints list
- GET /health : Health check (returns status/version)
- POST /chat : Send chat message to the assistant
- GET /session/{id} : Retrieve session history by session id

Example: send a message using curl

```powershell
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"message":"Show me an example of how to use the API"}'
```

Example response:

```json
{
	"response": "Sure — here is an example...",
	"session_id": "session_1",
	"intent": "ask_example"
}
```

To continue a session, include the returned `session_id` in subsequent requests:

```powershell
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"message":"Follow-up question","session_id":"session_1"}'
```

## Project layout

- `app.py` — FastAPI app & endpoints (entrypoint for the service)
- `rag_langchain.py` — RAG pipeline setup and retrieval/LLM wiring
- `graph.py` — State/graph orchestration used by the assistant
- `state.py` — Assistant state data structures and helpers
- `nodes_langchain.py` — Node implementations and LangChain adapters
- `chroma_langchain/` — Local Chroma DB and data directory (contains `chroma.sqlite3`)
- `plot.py`, `main.py` — utilities and example runners
- `requirements.txt` — Python dependencies
- `LICENSE` — Project license

## Notes & troubleshooting

- If the assistant fails to start, confirm provider API keys are set in environment variables.
- If you modify the retrieval DB, stop the server and restart so the pipeline re-initializes.
- Large models and embedding backends may require more memory and GPU — run on appropriate hardware when using big models.

## Contributing

Issues and PRs are welcome. When opening a PR, include a short description, reproduction steps, and tests if applicable.

## License

This project includes a `LICENSE` file — please review it for license details.