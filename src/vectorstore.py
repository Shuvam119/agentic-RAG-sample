"""Read access to THIS project's own vector index (vectorstore/chroma_db)."""

import logging
import re

import chromadb
import numpy as np

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PRODUCT_SIMILARITY_THRESHOLD,
    RELEVANCE_MARGIN,
    SIMILARITY_THRESHOLD,
    TOP_K,
    VECTORSTORE_DIR,
)
from src.embeddings import EmbeddingsGenerator

logger = logging.getLogger(__name__)

_client = None
_collection = None
_model = None
_stats_cache = None
_known_products = None


def get_vectorstore():
    """Return the process-wide ChromaDB client for this project's index."""
    global _client

    if _client is None:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    return _client


def reset_caches():
    """Drop cached client/collection/stats so a rebuild is picked up."""
    global _client, _collection, _stats_cache, _known_products

    _client = None
    _collection = None
    _stats_cache = None
    _known_products = None


def get_collection():
    """Return this project's collection, ensuring cosine distance."""
    global _collection

    if _collection is None:
        _collection = get_vectorstore().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    return _collection


def get_embeddings():
    """Return the shared embedding model (loaded lazily)."""
    global _model

    if _model is None:
        _model = EmbeddingsGenerator(model_name=EMBEDDING_MODEL)

    return _model


def embed_query(text):
    """Embed a query using the same BGE query-prefix convention as the index."""
    return get_embeddings().embed_text(text, is_query=True)


def _normalize_name(text):
    """Fold a name into a comparable form: lowercase, alphanumeric only."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def get_known_products():
    """Return the product names present in the index (excluding "General")."""
    global _known_products

    if _known_products is None:
        stats = get_stats()
        _known_products = [
            product for product in stats["products"]
            if product and product != "General"
        ]

    return _known_products


def query_mentions_known_product(query):
    """True when the query names a product that exists in the index.

    Normalizes both sides (StreamCutPro == "StreamCut Pro" == "stream-cut-pro")
    so a real-world product that happens to share the fictional product's name
    still routes to the internal documentation.
    """
    normalized = _normalize_name(query)

    if not normalized:
        return False

    return any(
        _normalize_name(product) in normalized
        for product in get_known_products()
    )


def retrieve(query, k=TOP_K, similarity_threshold=SIMILARITY_THRESHOLD):
    """Search the knowledge base and return only relevant chunks.

    Two gates decide whether a query is relevant to this corpus:

    * If the query explicitly names an indexed product (StreamCutPro,
      PolicyHub, ...) it is treated as a knowledge-base question and only the
      absolute floor PRODUCT_SIMILARITY_THRESHOLD applies. This guarantees a
      question about a product, even one that shares its name with a real-world
      product, returns documentation.
    * Otherwise the query must beat the corpus-mean similarity by a relevance
      margin, so unrelated queries (e.g. general knowledge) return an empty
      list instead of forcing the agent to cite irrelevant documentation.
    """

    collection = get_collection()
    count = collection.count()

    if not count:
        return []

    embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=count,
    )

    distances = results["distances"][0]
    similarities = [1 - distance for distance in distances]

    top_similarity = similarities[0]
    mean_similarity = sum(similarities) / len(similarities)

    mentions_product = query_mentions_known_product(query)

    if mentions_product:
        relevant = top_similarity >= PRODUCT_SIMILARITY_THRESHOLD
    else:
        relevant = (
            top_similarity >= similarity_threshold
            and (top_similarity - mean_similarity) >= RELEVANCE_MARGIN
        )

    if not relevant:
        return []

    documents = []

    for item_id, text, distance, metadata in zip(
        results["ids"][0][:k],
        results["documents"][0][:k],
        distances[:k],
        results["metadatas"][0][:k],
    ):
        documents.append(
            {
                "id": item_id,
                "text": text,
                "distance": distance,
                "similarity": 1 - distance,
                "metadata": metadata,
            }
        )

    return documents


def get_stats():
    """Return summary statistics about this project's knowledge base."""
    global _stats_cache

    if _stats_cache is not None:
        return _stats_cache

    collection = get_collection()
    count = collection.count()

    filenames = set()
    document_metadata = {}

    if count:
        all_metadata = collection.get(include=["metadatas"])

        for metadata in all_metadata["metadatas"]:
            if not metadata:
                continue

            filename = metadata.get("filename")

            if filename:
                filenames.add(filename)

                if filename not in document_metadata:
                    document_metadata[filename] = metadata

    result = {
        "collection": COLLECTION_NAME,
        "total_chunks": count,
        "document_count": len(filenames),
        "filenames": sorted(filenames),
        "products": sorted(
            {
                metadata.get("product", "General")
                for metadata in document_metadata.values()
            }
        ),
        "document_types": sorted(
            {
                metadata.get("document_type", "Unknown")
                for metadata in document_metadata.values()
            }
        ),
        "lifecycle_statuses": sorted(
            {
                metadata.get("lifecycle_status", "Fresh")
                for metadata in document_metadata.values()
            }
        ),
        "audiences": sorted(
            {
                metadata.get("audience", "End User")
                for metadata in document_metadata.values()
            }
        ),
        "versions": sorted(
            {
                metadata.get("version", "Unspecified")
                for metadata in document_metadata.values()
            }
        ),
        "vectorstore_path": str(VECTORSTORE_DIR),
    }

    _stats_cache = result
    return result
