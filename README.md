# CloudDesk Agentic RAG

An agentic retrieval-augmented generation (RAG) assistant for the fictional CloudDesk enterprise workspace. It answers questions from a local knowledge base of CloudDesk documentation using a Groq-hosted LLM, local BGE embeddings, and a ChromaDB vector store.

## Features

- **Agentic retrieval** - a LangChain agent decides when and how many times to search the knowledge base, refines queries, and combines evidence from multiple documents.
- **Metadata-aware search** - retrieved chunks carry version, document type, status (Current/Archived), and audience, so the agent can prefer current docs and resolve version conflicts.
- **Streamlit chat UI** - with full chat history and document badges in the sidebar.
- **Fast startup** - the UI renders immediately; the LLM, embeddings model, and vector store load lazily on the first question and are cached for the session.

## Architecture

```
ui/app.py            Streamlit chat interface (lazy agent initialization)
src/agent.py         Agent construction + system prompt (13 rules)
src/rag_tool.py      search_knowledge_base tool (k=8, metadata-aware)
src/vectorstore.py   ChromaDB persistence (chroma_db/)
src/embeddings.py    BAAI/bge-small-en-v1.5 embeddings
src/ingestion.py     Load/split DOCX/PDF docs + extract metadata
src/indexer.py       Build the vector index (run via python -m src.indexer)
data/raw/            8 sample CloudDesk documents (.docx)
```

- **LLM:** `llama-3.1-8b-instant` via Groq (temperature 0)
- **Embeddings:** `BAAI/bge-small-en-v1.5` (local, 384 dims)
- **Vector store:** ChromaDB on disk (`chroma_db/`)
- **Orchestration:** LangChain `create_agent` (LangGraph) with a custom `search_knowledge_base` tool

## Setup

Requirements: Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create `.env` (the app reads `GROQ_API_KEY`):

```
GROQ_API_KEY=your_groq_api_key
```

Build the vector index (existing index is reused automatically if present):

```bash
python -m src.indexer
```

Note: use `python -m src.indexer`, not `python src/indexer.py`.

## Run the app

```bash
streamlit run ui/app.py
```

Open http://localhost:8501 and ask questions such as:

- How long are CloudDesk API v2 access tokens valid?
- What is the recommended version and why?
- How do I resolve 401 errors in the v2 API?
- What changed for users between v1 and v2?

## Manual checks

- `test_agent.py` - interactive agent Q&A in the terminal
- `test_retrieval.py` - prints retrieved chunks with metadata

## Notes

- The project uses the LangChain 1.x `create_agent` API (LangGraph-based); the legacy `create_tool_calling_agent`/`AgentExecutor` API is not available in this version.
- `chroma_db/` is rebuilt only by `python -m src.indexer`; the app only loads the existing index.
- `test_agent.py` and `test_retrieval.py` prompt for input via stdin.
