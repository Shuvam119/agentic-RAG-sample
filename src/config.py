"""Application configuration and constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "chroma_db"

# Source directory for the raw documents indexed by this project.
# Defaults to the Traditional RAG project's document folder because both
# projects share the same document set; this project builds its OWN index.
DEFAULT_DOCS_SOURCE = Path(r"C:\Users\USER\ai-docs-copilot\data\raw")
DOCS_SOURCE = Path(os.getenv("DOCS_SOURCE", str(DEFAULT_DOCS_SOURCE)))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION_NAME = "documents"

# Relevance gate for retrieved results. A query is only considered relevant
# if its top hit is at least SIMILARITY_THRESHOLD similar (absolute) AND beats
# the corpus-mean similarity by at least RELEVANCE_MARGIN. Unrelated queries
# (e.g. general knowledge) therefore return nothing instead of forcing the
# agent to cite irrelevant documentation.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
RELEVANCE_MARGIN = float(os.getenv("RELEVANCE_MARGIN", "0.10"))

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
