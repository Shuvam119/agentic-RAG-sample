"""Embeddings generator using sentence-transformers (BGE query/passage prefixes)."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    """Generates embeddings for text chunks and queries."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def _prepare_text(self, text: str, *, is_query: bool) -> str:
        """Apply the BGE query/passage prefix convention."""
        if "bge" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            return prefix + text
        return text

    def embed_text(self, text: str, *, is_query: bool = False) -> np.ndarray:
        prepared = self._prepare_text(text, is_query=is_query)
        return self.model.encode(prepared, convert_to_numpy=True)

    def embed_texts(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        prepared = [self._prepare_text(t, is_query=is_query) for t in texts]
        return self.model.encode(prepared, convert_to_numpy=True)

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(chunk_texts, is_query=False)

        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy["embedding"] = embedding
            chunks_with_embeddings.append(chunk_copy)

        return chunks_with_embeddings
