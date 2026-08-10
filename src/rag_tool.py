from langchain_core.tools import tool

from src.vectorstore import retrieve


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal documentation knowledge base.

    Use this tool when the user's question requires
    information from the indexed internal documents.
    """

    try:
        documents = retrieve(query, k=8)

    except Exception:
        return (
            "The internal knowledge base is currently unavailable. "
            "No documentation was retrieved."
        )

    if not documents:
        return "No relevant documentation was found."

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
