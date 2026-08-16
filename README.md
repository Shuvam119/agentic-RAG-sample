# Agentic RAG

An agentic retrieval-augmented generation (RAG) assistant. It answers
questions primarily from an **internal documentation knowledge base** and uses
live web search (via Tavily) only as a fallback, using a Groq-hosted LLM and
local BGE embeddings.

## Knowledge base

- This is a **standalone project with its own vector index**. It is not an
  extension of the Traditional RAG project.
- The indexed documents live in this project's own `data/raw` folder
  (configurable via `DOCS_SOURCE`) and are indexed into this project's own
  ChromaDB at `vectorstore/chroma_db`.
- It never writes to, reads from, or depends on the Traditional RAG project's
  index.

## How the agent decides what to search

The behavior is deliberately simple and knowledge-base-first:

1. **The knowledge base is always searched first.** For every question the
   agent calls `search_knowledge_base` before anything else - there are no
   exceptions and the agent never skips it in favor of the web.
2. **Documentation wins.** If the knowledge base returns sources, the answer
   is built from those documents and cites them (`[SOURCE N]`).
3. **Product questions always get documentation answers.** The products in
   the docs (e.g. StreamCutPro, PolicyHub) are the company's own. Even if the
   question is about a real-world product that happens to share the same name,
   the agent answers from the internal documentation - and may add a one-line
   anecdote noting that a similarly named real-world product exists.
4. **Web search is only a fallback.** `search_web` is used only when the
   knowledge base returned `NO_RELEVANT_DOCUMENTATION` and the question is
   genuinely general knowledge, current events, or external information
   (e.g. "What is the capital of India?"). It is never used to answer product
   questions.

## Features

- **Knowledge-base-first agent** - deterministic routing: the docs are always
  consulted first, web search is a fallback only, and invented sources are
  prohibited.
- **Relevance-gated retrieval** - chunks below a cosine-similarity threshold
  are dropped, and the query must also share a meaningful term with the
  retrieved chunks, so unrelated general-knowledge questions do not get forced
  documentation sources.
- **Product-name boost** - when a query explicitly names an indexed product
  (StreamCutPro, PolicyHub), a relaxed similarity floor applies so the query
  always routes to the knowledge base even when the fictional product name
  matches a real-world product.
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
src/agent.py         Agent construction + KB-first system prompt
src/rag_tool.py      search_knowledge_base tool (k=TOP_K, metadata-aware)
src/web_search_tool.py  search_web tool (Tavily, fallback only)
src/vectorstore.py   Read access to this project's own ChromaDB + relevance gate
                     (product-name boost in query_mentions_known_product)
src/embeddings.py    BAAI/bge-small-en-v1.5 embeddings (BGE query/passage prefixes)
src/ingestion.py     Load DOCX/PDF, extract metadata, version-aware lifecycle, chunk
src/indexer.py       Build this project's index (python -m src.indexer)
src/config.py        Paths, model, thresholds
```

- **LLM:** `openai/gpt-oss-120b` via Groq (temperature 0).
  Note: `llama-3.3-70b-versatile` is **not** recommended - Groq's tool-calling
  for that model is unreliable when two tools are bound.
- **Embeddings:** `BAAI/bge-small-en-v1.5` (local, 384 dims)
- **Vector store:** this project's ChromaDB (`vectorstore/chroma_db`,
  collection `documents`, cosine distance)
- **Web search:** Tavily (`search_depth="basic"`, up to 3 results)
- **Orchestration:** LangChain `create_agent` (LangGraph) with two tools:
  `search_knowledge_base` and `search_web`

## Setup

Requirements: Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create `.env` (the app reads `GROQ_API_KEY`, `TAVILY_API_KEY`,
`MODEL_NAME`, and `DOCS_SOURCE`):

```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
MODEL_NAME=openai/gpt-oss-120b
DOCS_SOURCE=C:\Users\USER\agentic-rag\data\raw
```

- `TAVILY_API_KEY` is optional - if missing, `search_web` returns a
  "not configured" message and the agent answers from the internal
  knowledge base only.
- `DOCS_SOURCE` defaults to this project's own `data/raw` folder. Point it
  anywhere; the documents are indexed into this project's own vector store.

Build this project's index (run once and after documents change):

```bash
python -m src.indexer
```

## Run the app

```bash
streamlit run ui/app.py
```

Open http://localhost:8501 and ask questions such as:

- How do I render and export video in StreamCutPro? *(internal)*
- I found a real product called StreamCut Pro online - what is it? *(internal,
  with an anecdote about the similarly named real product)*
- How do I integrate the StreamCutPro API? *(internal)*
- What is the parental leave policy? *(internal)*
- What is the capital of India? *(web fallback - the knowledge base is still
  checked first)*

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
