"""Read access to THIS project's own vector index (vectorstore/chroma_db)."""

import json
import logging
import re
import threading

import numpy as np

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PRODUCT_SIMILARITY_THRESHOLD,
    SIMILARITY_THRESHOLD,
    TOP_K,
    VECTORSTORE_DIR,
)
from src.embeddings import EmbeddingsGenerator

logger = logging.getLogger(__name__)

_client = None
_collection = None
_model = None
_client_lock = threading.Lock()
_collection_lock = threading.Lock()
_model_lock = threading.Lock()
_stats_cache = None
_stats_cache_sig = None
_stats_lock = threading.Lock()
_stats_path = VECTORSTORE_DIR / "stats.json"
_known_products = None


def get_vectorstore():
    """Return the process-wide ChromaDB client for this project's index."""
    global _client

    if _client is None:
        # chromadb is slow to import (onnxruntime/tokenizers), so it is loaded
        # lazily and only when a collection is actually needed.
        with _client_lock:
            if _client is None:
                import chromadb

                VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
                _client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    return _client


def reset_caches():
    """Drop cached client/collection/stats so a rebuild is picked up."""
    global _client, _collection, _stats_cache, _stats_cache_sig, _known_products

    _client = None
    _collection = None
    _stats_cache = None
    _stats_cache_sig = None
    _known_products = None

    if _stats_path.exists():
        try:
            _stats_path.unlink()
        except OSError:
            pass


def get_collection():
    """Return this project's collection, ensuring cosine distance."""
    global _collection

    if _collection is None:
        with _collection_lock:
            if _collection is None:
                _collection = get_vectorstore().get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )

    return _collection


def get_embeddings():
    """Return the shared embedding model (loaded lazily, thread-safe)."""
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = EmbeddingsGenerator(model_name=EMBEDDING_MODEL)

    return _model


def embed_query(text):
    """Embed a query using the same BGE query-prefix convention as the index."""
    return get_embeddings().embed_text(text, is_query=True)


def _normalize_name(text):
    """Fold a name into a comparable form: lowercase, alphanumeric only."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


# Stopwords are too common to signal that a chunk is on-topic, so they are
# ignored when checking whether a query's terms appear in the retrieved text.
_STOPWORDS = frozenset({
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "get", "got", "had",
    "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "should", "so", "that", "the", "their",
    "there", "these", "they", "this", "those", "to", "use", "using", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your",
})


def _query_terms(query):
    """Meaningful content-word tokens of a query, stopwords removed."""
    return [
        word
        for word in re.findall(r"[a-z]+", query.lower())
        if len(word) >= 3 and word not in _STOPWORDS
    ]


def _query_terms_overlap(query, texts):
    """True when a query term actually appears in the retrieved texts.

    This lexical check keeps the embedding gate honest. A query about a topic
    the docs never mention (e.g. "what is streamlit") can score highly against
    a product description purely because of the shared "what is X / X is"
    pattern, but it shares no vocabulary with the corpus and is therefore
    rejected instead of forcing the agent to cite irrelevant documentation.
    Morphological variants (documentation/documents, setting/settings) are
    treated as the same term via containment and shared-prefix matching.
    """
    terms = _query_terms(query)

    if not terms:
        return False

    chunk_words = [
        word
        for text in texts
        for word in re.findall(r"[a-z]+", text.lower())
        if len(word) >= 3
    ]

    for term in terms:
        for word in chunk_words:
            if term == word:
                return True
            if len(term) >= 4 and term in word:
                return True
            if len(word) >= 4 and word in term:
                return True

            shared = 0
            for a, b in zip(term, word):
                if a != b:
                    break
                shared += 1
            if shared >= 7 and len(term) >= 7 and len(word) >= 7:
                return True

    return False


def get_known_products():
    """Return the product names present in the index (excluding "General")."""
    global _known_products

    stats = get_stats()
    products = [
        product for product in stats["products"]
        if product and product != "General"
    ]

    if _known_products != products:
        _known_products = products

    return _known_products


_PRODUCT_PREFIX_MIN_LEN = 6
_PRODUCT_COVERAGE_RATIO = 0.6


def _query_mentions_product(query_normalized, product_normalized):
    """True when the normalized query mentions the product.

    Matches a full product name ("StreamCutPro") as well as a long-enough
    contiguous prefix of it ("streamcut"), so generic questions like "what is
    streamcut" still route to the internal documentation. Prefixes below
    _PRODUCT_PREFIX_MIN_LEN or covering less than _PRODUCT_COVERAGE_RATIO of
    the name (e.g. "stream") are ignored to avoid false positives.
    """
    if product_normalized in query_normalized:
        return True

    max_len = len(product_normalized)

    if max_len <= _PRODUCT_PREFIX_MIN_LEN:
        return False

    for length in range(max_len - 1, _PRODUCT_PREFIX_MIN_LEN - 1, -1):
        prefix = product_normalized[:length]
        if (
            prefix in query_normalized
            and length / max_len >= _PRODUCT_COVERAGE_RATIO
        ):
            return True

    return False


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
        _query_mentions_product(normalized, _normalize_name(product))
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
    * Otherwise the query must meet the absolute SIMILARITY_THRESHOLD and share
      at least one meaningful term with the retrieved chunks
      (_query_terms_overlap). Queries that only pattern-match (e.g. general
      knowledge such as "what is streamlit") score well on embeddings but share
      no vocabulary with the corpus, so they return an empty list instead of
      forcing the agent to cite irrelevant documentation.
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

    mentions_product = query_mentions_known_product(query)

    if mentions_product:
        relevant = top_similarity >= PRODUCT_SIMILARITY_THRESHOLD
    else:
        top_texts = results["documents"][0][:k]
        relevant = (
            top_similarity >= similarity_threshold
            and _query_terms_overlap(query, top_texts)
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
    """Return summary statistics about this project's knowledge base.

    Fast path: a JSON sidecar written at index time (and refreshed after each
    rebuild) is read directly, so the UI never has to load chromadb just to
    display sidebar stats. The cached value is invalidated whenever the sidecar
    changes on disk, so a rebuilt index is picked up without restarting the
    app. Falls back to computing from the collection, which refreshes the
    sidecar.
    """
    global _stats_cache, _stats_cache_sig

    signature = _sidecar_signature()

    if _stats_cache is not None and _stats_cache_sig == signature:
        return _stats_cache

    with _stats_lock:
        if _stats_cache is not None and _stats_cache_sig == signature:
            return _stats_cache

        cached = _read_stats_sidecar()
        if cached is not None:
            _stats_cache = cached
            _stats_cache_sig = signature
            return _stats_cache

        result = _compute_stats()
        _stats_cache = result
        _stats_cache_sig = signature
        _write_stats_sidecar(result)

    return _stats_cache


def _sidecar_signature():
    """Signature of the stats sidecar, so cached stats can detect a rebuild.

    Returns None when the sidecar does not exist (e.g. the index has never
    been built or was deleted).
    """
    try:
        stat = _stats_path.stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


def _read_stats_sidecar():
    try:
        if _stats_path.exists():
            with _stats_path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass

    return None


def _write_stats_sidecar(result):
    try:
        _stats_path.parent.mkdir(parents=True, exist_ok=True)
        with _stats_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _compute_stats():
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

    return {
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
