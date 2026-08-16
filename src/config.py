"""Application configuration and constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "chroma_db"

# Source directory for the raw documents indexed by this project.
# This project is self-contained: it indexes the documents in its own
# data/raw folder into its own vectorstore.
DEFAULT_DOCS_SOURCE = PROJECT_ROOT / "data" / "raw"
DOCS_SOURCE = Path(os.getenv("DOCS_SOURCE", str(DEFAULT_DOCS_SOURCE)))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION_NAME = "documents"

# Relevance gate for retrieved results. A query is only considered relevant
# if its top hit is at least SIMILARITY_THRESHOLD similar (absolute) AND shares
# at least one meaningful term with the retrieved chunks. Unrelated queries
# (e.g. general knowledge) therefore return nothing instead of forcing the
# agent to cite irrelevant documentation.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))

# Relaxed absolute-similarity floor applied when the query explicitly names a
# product that exists in the index (e.g. StreamCutPro, PolicyHub). This keeps
# product questions in the knowledge base even when the fictional product name
# happens to match a real-world product with the same name.
PRODUCT_SIMILARITY_THRESHOLD = float(
    os.getenv("PRODUCT_SIMILARITY_THRESHOLD", "0.50"))

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
