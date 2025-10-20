# RAG Code Assistant API

AI-powered code generation and explanation assistant using LangGraph and LangChain.

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the API
```bash
python -m app.main
# or
uvicorn app.main_app:app --reload
```

### 3. Access the API
- **Swagger UI:** http://localhost:8000/docs  
- **ReDoc:** http://localhost:8000/redoc  
- **API root:** http://localhost:8000

## 📡 API Endpoints

### `GET /` — Root
Check API status

### `GET /health` — Health check
Health status of the service

### `GET /examples` — Get examples
List of example queries

### `POST /query` — Process query
Auto-detect intent and process query
```bash
curl -X POST "http://localhost:8000/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "Generate a factorial function"}'
```

### `POST /generate` — Generate code
Force code generation
```bash
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{"query": "Create a binary search function"}'
```

### `POST /explain` — Explain code
Force code explanation
```bash
curl -X POST "http://localhost:8000/explain" \
    -H "Content-Type: application/json" \
    -d '{"query": "How does quicksort work?"}'
```

## 🏗️ Architecture

User Request → FastAPI → LangGraph State Machine  
Router Node  
├─ Generate Code → RAG Chain → Response  
└─ Explain Code  → RAG Chain → Response

(Visualize with plot.py)

## 📁 Project Structure

- `app/main.py` — FastAPI application  
- `app/config.py` — Configuration  
- `app/models.py` — Pydantic models  
- `graph.py` — LangGraph workflow  
- `nodes_langchain.py` — Graph nodes  
- `rag_langchain.py` — RAG setup  
- `state.py` — State definition  
- `plot.py` — Graph visualization

## 🔧 Configuration

Edit the `.env` file to customize settings.

## 📊 Visualization

Generate graph visualization:
```bash
python plot.py
```
