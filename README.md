# AI Assistant with Configurable RAG, Hybrid Search & Conversation Memory

A full-stack AI Assistant built with FastAPI, LangChain, Google Gemini, Qdrant Cloud Hybrid Search (gemini-embedding-2 dense 3072 + FastEmbed BM25 sparse), BGE-Reranker-v2-M3, Tavily Web Search, PostgreSQL, and a minimalist black-and-white React 18 frontend.

---

## Key Architecture & Features

1. **LLM Orchestration**:
   - Google Gemini via `langchain-google-genai`.
   - Per-message selectable thinking levels: `low` (1,024 tokens), `med` (4,096 tokens), and `high` (16,384 tokens).
   - Autonomous agent tools:
     - `add_to_memory(fact)`: Stores cross-conversation facts in PostgreSQL.
     - `remove_from_memory(fact)`: Hard-deletes matching stored facts.
     - `clear_conversation_history()`: Hard-deletes all messages in the current conversation thread.
     - `tavily_search(query)`: Autonomous web search tool.

2. **Conversation Modes**:
   - **Linear Mode**: Standard chat thread with continuous message history reconstructed from PostgreSQL on every request. Supports inline message editing and hard deletion.
   - **Isolated Mode**: Distinct UI with independent question cards. Questions have no automatic history by default, with an explicit **Exchange Picker** allowing the user to link a question to exactly one prior `(hk, ak)` exchange.

3. **Configurable RAG (Hybrid Search & Reranking)**:
   - **Dense + Sparse Embeddings**: Google `gemini-embedding-2` generating 3072-dim dense vectors, combined with `fastembed` BM25 sparse lexical tokens.
   - **Vector DB**: Qdrant Cloud cluster with named dense (`dense`, 3072, Cosine) and sparse vectors (`sparse`, `idf` modifier), payload index on `user-id` (tenant index), searched via Reciprocal Rank Fusion (RRF).
   - **Cross-Encoder Reranker**: `BAAI/bge-reranker-v2-m3` reranks `top_k` retrieved candidates to retain the `top_n` most relevant passages.
   - Always-visible **RAG toggle** (ON/OFF) changeable mid-conversation.

4. **Persistent User Memory**:
   - Dedicated PostgreSQL `memory` table storing persistent facts across conversations.
   - Always-visible **Memory toggle** (ON/OFF).
   - UI modal to view, manually add, and delete stored facts.

5. **Centralized Configuration**:
   - All tunable parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `TOP_N`, model names, temperatures, Qdrant collection, thinking budgets) live in [backend/config.py](file:///c:/Users/Za7lou9/Desktop/ai%20assistant/backend/config.py).

---

## Directory Structure

```
ai assistant/
├── backend/
│   ├── agent/
│   │   ├── orchestrator.py      # LLM deliberation, tool loop, context rebuild
│   │   └── tools.py             # LangChain agent tools (memory, history, Tavily)
│   ├── rag/
│   │   ├── embeddings.py        # BGE-M3 (dense+sparse) & FlagReranker
│   │   ├── ingestion.py         # Text chunking and document parser (.txt, .md, .pdf)
│   │   ├── retrieval.py         # Hybrid search + reranker pipeline
│   │   └── vector_store.py      # Qdrant client & RRF hybrid indexing
│   ├── routers/
│   │   ├── config_route.py      # Non-secret runtime settings
│   │   ├── conversations.py     # Conversation CRUD (hard delete cascade)
│   │   ├── memory.py            # Persistent user memory CRUD
│   │   ├── messages.py          # Send, edit (cascade delete subsequent), delete
│   │   └── rag.py               # Document ingestion & stats
│   ├── config.py                # Centralized configuration (Pydantic Settings)
│   ├── database.py              # PostgreSQL connection & session factory
│   ├── main.py                  # FastAPI application entrypoint
│   ├── models.py                # PostgreSQL SQLAlchemy models
│   └── schemas.py               # Pydantic request & response schemas
├── frontend/
│   ├── src/
│   │   ├── api/client.js        # Centralized Axios client
│   │   ├── components/
│   │   │   ├── HeaderBar.jsx    # Toggles (RAG, Memory, Thinking level)
│   │   │   ├── IsolatedChatView.jsx # Distinct UI for isolated exchanges
│   │   │   ├── KnowledgeModal.jsx   # RAG upload and ingestion UI
│   │   │   ├── LinearChatView.jsx   # Linear chat thread with inline edits
│   │   │   ├── MemoryModal.jsx      # Persistent facts viewer/manager
│   │   │   ├── MessageInput.jsx     # Input box + exchange picker
│   │   │   └── Sidebar.jsx          # Conversation list & new chat mode
│   │   ├── hooks/useLocalStorage.js # LocalStorage sync (drafts, last conv, toggles)
│   │   ├── App.jsx              # Application shell
│   │   └── index.css            # Modern black-and-white minimalist CSS theme
│   ├── package.json             # React 18, Vite, Axios
│   └── vite.config.js           # Proxy config to port 8000
├── app_tests/
│   └── test_app.py              # Verification test suite
├── ingest.py                    # Standalone CLI document ingestion tool
├── example.env                  # Credential and parameter template
└── .env                         # Local development configuration
```

---

## Setup & Running

### 1. Prerequisites
- **Python 3.10+** (tested on Python 3.12)
- **Node.js 18+** & **npm**
- **PostgreSQL 14+** running locally (e.g. `localhost:5432`)

### 2. Environment Configuration
Copy `example.env` to `.env` (or configure the existing `.env`):
```bash
cp example.env .env
```
Fill in your API credentials:
```env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
DATABASE_URL=postgresql://postgres:password@localhost:5432/ai_assistant

# Qdrant Cloud Configuration
QDRANT_ENDPOINT=https://your-cluster-id.us-east-1.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_cloud_api_key_here
QDRANT_COLLECTION_NAME=ai-assistant-collection
```
*Note: If `GOOGLE_API_KEY` is not provided, the server will still run and return a friendly configuration notice upon sending messages.*

### 3. Standalone Document Ingestion
You can ingest individual documents or whole directories directly via CLI into Qdrant:
```bash
# Ingest a single file (.txt, .md, .pdf, .py, etc.)
python ingest.py --file path/to/document.pdf

# Ingest an entire folder
python ingest.py --path ./docs
```

### 4. Running the Application

#### Start Backend (FastAPI)
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be accessible at `http://127.0.0.1:8000/docs`.

#### Start Frontend (React 18 + Vite)
In a separate terminal:
```bash
cd frontend
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## Running Automated Tests

Run the built-in test suite verifying config, database cascade deletions, isolated exchange linking, message editing, agent tools, and Qdrant hybrid retrieval:
```bash
python app_tests/test_app.py
```
