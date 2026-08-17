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

The agent combines **code-enforced routing** (deterministic) with an
**intelligent LLM** that can decide when web search adds value:

1. **Greetings are caught in code.** "hi", "hello", "thanks", etc. are
   detected by a regex guard and answered directly — no tools, no KB, no web.
2. **The knowledge base is always searched first.** Every substantive question
   enters the `search_knowledge_base` node before anything else. There are no
   exceptions and the LLM is never given the choice to skip it.
3. **The LLM decides whether web search is needed.** After receiving KB
   results, the answering LLM has `search_web` available as a tool and uses it
   intelligently:
   - **KB answers the question** → the LLM answers from documentation and
     cites `[SOURCE N]`. It does *not* call search_web.
   - **KB has docs but not the answer** (e.g. "what are its market
     equivalents?" when the KB only describes the product) → the LLM answers
     what it can from the docs, then calls `search_web` for the rest.
   - **KB has nothing** → the LLM decides: external / current-events
     questions get `search_web`; self-contained facts and greetings do not.
4. **Web results are synthesized without tools.** After executing `search_web`,
   the agent composes a final answer from both KB and web sources using a
   no-tools LLM call, so it cannot make additional web requests.

## Features

- **Knowledge-base-first agent** - KB is always consulted first; web search
  is used only when the LLM judges it necessary for a complete answer.
- **Greeting guard** - "hi", "hello", etc. are answered instantly with no
  tools, no KB lookup, and no web search.
- **Intelligent hybrid search** - the LLM combines KB documentation with web
  results when the documentation alone is insufficient.
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
src/agent.py         LangGraph agent: greeting guard → KB-first → LLM-decides web
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
- **Orchestration:** a hand-built LangGraph (`StateGraph`) with a greeting
  guard, KB-first path, and an LLM that decides whether to call `search_web`
  when the documentation is insufficient.

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

- The agent is a hand-built LangGraph `StateGraph` (LangGraph 1.x), not the
  LangChain `create_agent`/`AgentExecutor` API. The graph exposes the same
  `invoke`/`stream` contract, so the UI and `test_agent.py` are unchanged.
- `vectorstore/chroma_db` is rebuilt only by `python -m src.indexer`.
- `test_agent.py` and `test_retrieval.py` prompt for input via stdin.
