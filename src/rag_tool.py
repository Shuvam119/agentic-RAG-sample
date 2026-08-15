from langchain_core.tools import tool

from src.config import TOP_K
from src.vectorstore import retrieve


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal documentation knowledge base.

    Call this tool first for every question. Returns documentation sources or NO_RELEVANT_DOCUMENTATION.
    """
    try:
        documents = retrieve(query, k=TOP_K)

    except Exception:
        return (
            "NO_RELEVANT_DOCUMENTATION. "
            "The internal knowledge base is currently unavailable. "
            "No documentation was retrieved. There are no documentation "
            "sources to cite."
        )

    if not documents:
        return (
            "NO_RELEVANT_DOCUMENTATION. The internal knowledge base returned "
            "no documentation matching this query. There are ZERO documentation "
            "sources, so nothing can be cited from the knowledge base. If you "
            "still need an answer, use general knowledge or search_web."
        )

    results = []

    for i, document in enumerate(documents, 1):

        metadata = document["metadata"]

        results.append(
            f"""
SOURCE {i}
Title: {metadata.get("title") or metadata.get("filename", "Unknown")}
Version: {metadata.get("version", "Unknown")}
Document Type: {metadata.get("document_type", "Unknown")}
Status: {metadata.get("lifecycle_status", "Unknown")}
Audience: {metadata.get("audience", "Unknown")}
Product: {metadata.get("product", "Unknown")}
Source: {metadata.get("filename", "Unknown")}

Content:
{document["text"]}
"""
        )

    return "\n".join(results)
