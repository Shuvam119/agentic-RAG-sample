# Agentic RAG

An agentic retrieval-augmented generation (RAG) assistant. It answers
questions by combining an internal documentation knowledge base with live web
search (via Tavily), using a Groq-hosted LLM and local BGE embeddings.

## Knowledge base

- This is a **standalone project with its own vector index**. It is not an
  extension of the Traditional RAG project.
- The indexed documents are the same set the Traditional RAG project uses
  (`C:\Users\USER\ai-docs-copilot\data\raw`, configurable via `DOCS_SOURCE`),
  but this project **builds and maintains its own ChromaDB** at
  `vectorstore/chroma_db`.
- It never writes to, reads from, or depends on the Traditional RAG project's
  index.

## Features

- **Dual-source agent** - the agent chooses between `search_knowledge_base`
  (internal docs) and `search_web` (public web), and can combine both for
  hybrid research questions.
- **Relevance-gated retrieval** - chunks below a cosine-similarity threshold
  are dropped, so general-knowledge questions do not get forced documentation
  sources.
- **Agentic retrieval** - the agent decides when and how many times to
  search, refines queries, and combines evidence from multiple documents.
- **Metadata-aware search** - retrieved chunks carry version, document type,
  status, product, and audience, so the agent can prefer current docs and
  resolve version conflicts.
- **Streamlit chat UI** - with full chat history and a sidebar that lists
  the products, document types, statuses, audiences, and versions present
  in the internal knowledge base.
- **Fast startup** - the UI renders immediately; the LLM, embeddings model,
  and vector store load lazily on the first question and are cached for the
  session.

## Architecture

```
ui/app.py            Streamlit chat interface (streams agent steps, shows sources)
src/agent.py         Agent construction + system prompt (source-selection rules)
src/rag_tool.py      search_knowledge_base tool (k=8, metadata-aware)
src/web_search_tool.py  search_web tool (Tavily, 5 results)
src/vectorstore.py   Read access to this project's own ChromaDB + relevance gate
src/embeddings.py    BAAI/bge-small-en-v1.5 embeddings (BGE query/passage prefixes)
src/ingestion.py     Load DOCX/PDF, extract metadata, version-aware lifecycle, chunk
src/indexer.py       Build this project's index (python -m src.indexer)
src/config.py        Paths, model, thresholds
```

- **LLM:** `llama-3.1-8b-instant` via Groq (temperature 0)
- **Embeddings:** `BAAI/bge-small-en-v1.5` (local, 384 dims)
- **Vector store:** this project's ChromaDB (`vectorstore/chroma_db`,
  collection `documents`, cosine distance)
- **Web search:** Tavily (`search_depth="basic"`, up to 5 results)
- **Orchestration:** LangChain `create_agent` (LangGraph) with two tools:
  `search_knowledge_base` and `search_web`

## Setup

Requirements: Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create `.env` (the app reads `GROQ_API_KEY`, `TAVILY_API_KEY`, and
`DOCS_SOURCE`):

```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
DOCS_SOURCE=C:\Users\USER\ai-docs-copilot\data\raw
```

- `TAVILY_API_KEY` is optional - if missing, `search_web` returns a
  "not configured" message and the agent answers from the internal
  knowledge base only.
- `DOCS_SOURCE` defaults to the Traditional RAG project's `data/raw`
  folder. Point it anywhere; the documents are indexed into this project's
  own vector store.

Build this project's index (run once and after documents change):

```bash
python -m src.indexer
```

## Run the app

```bash
streamlit run ui/app.py
```

Open http://localhost:8501 and ask questions such as:

- What do the internal documents say about the products covered?
  *(internal)*
- What is the recommended version of a product and why?
  *(internal)*
- What are the latest developments in OAuth security? *(web)*
- What does the documentation say about authentication, and what are
  the latest OAuth security recommendations? *(hybrid)*

General-knowledge questions (e.g. "What is the capital of India?") are
answered without searching the documentation.

## Manual checks

- `test_agent.py` - interactive agent Q&A in the terminal
- `test_retrieval.py` - prints retrieved chunks with metadata
- `test_web_search.py` - interactive single web search via Tavily

## Notes

- The project uses the LangChain 1.x `create_agent` API (LangGraph-based);
  the legacy `create_tool_calling_agent`/`AgentExecutor` API is not
  available in this version.
- `vectorstore/chroma_db` is rebuilt only by `python -m src.indexer`.
- `test_agent.py` and `test_retrieval.py` prompt for input via stdin.
