"""Build this project's own vector index from the raw documents.

Run:  python -m src.indexer

The documents are read from DOCS_SOURCE (default: the Traditional RAG
project's data/raw folder) and indexed into THIS project's ChromaDB at
vectorstore/chroma_db. The Traditional RAG project is never modified.
"""

import sys

from src.config import COLLECTION_NAME, VECTORSTORE_DIR
from src.embeddings import EmbeddingsGenerator
from src.ingestion import (
    apply_version_lifecycle,
    extract_metadata,
    load_documents_from_directory,
    DocumentChunker,
)
from src.vectorstore import get_collection, get_vectorstore, reset_caches


def build(rebuild: bool = True) -> int:
    documents = load_documents_from_directory()

    if not documents:
        print("No PDF or DOCX documents found in the docs source directory.")
        return 1

    for document in documents:
        document["metadata"].update(extract_metadata(document))

    apply_version_lifecycle(documents)

    print(f"Loaded {len(documents)} document(s):")
    for document in documents:
        metadata = document["metadata"]
        print(
            f"  - {metadata['filename']} | {metadata['document_type']} | "
            f"v{metadata['version']} | {metadata['lifecycle_status']}"
        )

    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunk(s).")

    embedder = EmbeddingsGenerator()
    chunks_with_embeddings = embedder.embed_chunks(chunks)
    print("Embeddings generated.")

    if rebuild:
        client = get_vectorstore()
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        reset_caches()

    collection = get_collection()

    ids = []
    embeddings = []
    texts = []
    metadatas = []

    for chunk in chunks_with_embeddings:
        metadata = chunk["metadata"]
        ids.append(metadata["chunk_id"])
        embeddings.append(chunk["embedding"].tolist())
        texts.append(chunk["text"])
        metadatas.append({
            "filename": metadata["filename"],
            "chunk_id": metadata["chunk_id"],
            "document_type": metadata["document_type"],
            "chunk_index": metadata["chunk_index"],
            "total_chunks": metadata["total_chunks"],
            "source": metadata.get("source", ""),
            "title": metadata.get("title", metadata["filename"]),
            "product": metadata.get("product", "General"),
            "version": metadata.get("version", "Unspecified"),
            "audience": metadata.get("audience", "End User"),
            "department": metadata.get("department", "Documentation"),
            "author": metadata.get("author", "Unknown"),
            "last_updated": metadata.get("last_updated", ""),
            "publication_date": metadata.get("publication_date", ""),
            "lifecycle_status": metadata.get("lifecycle_status", "Fresh"),
            "keywords": ", ".join(metadata.get("keywords", [])),
            "summary": metadata.get("summary", ""),
        })

    collection.add(ids=ids, embeddings=embeddings, documents=texts,
                   metadatas=metadatas)

    print(f"Indexed {len(ids)} chunk(s) into {VECTORSTORE_DIR} "
          f"(collection: {COLLECTION_NAME}).")
    return 0


if __name__ == "__main__":
    sys.exit(build())
