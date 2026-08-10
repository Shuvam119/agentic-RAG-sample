"""Read access to THIS project's own vector index (vectorstore/chroma_db)."""

import logging

import chromadb
import numpy as np

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
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


def get_vectorstore():
    """Return the process-wide ChromaDB client for this project's index."""
    global _client

    if _client is None:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    return _client


def reset_caches():
    """Drop cached client/collection/stats so a rebuild is picked up."""
    global _client, _collection, _stats_cache

    _client = None
    _collection = None
    _stats_cache = None


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


def retrieve(query, k=TOP_K, similarity_threshold=SIMILARITY_THRESHOLD):
    """Search the knowledge base and return only relevant chunks.

    A query must beat the corpus-mean similarity by a relevance margin before
    any chunk is returned, so an unrelated query (e.g. general knowledge)
    returns an empty list instead of irrelevant documentation.
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

    if (
        top_similarity < similarity_threshold
        or (top_similarity - mean_similarity) < RELEVANCE_MARGIN
    ):
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
