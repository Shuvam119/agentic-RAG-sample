from langchain_core.tools import tool

from src.vectorstore import get_vectorstore


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the CloudDesk documentation.

    Use this tool when the user's question requires
    information from CloudDesk documentation.
    """

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
    )

    documents = retriever.invoke(query)

    if not documents:
        return "No relevant documentation was found."

    results = []

    for i, document in enumerate(documents, 1):

        metadata = document.metadata

        results.append(
            f"""
SOURCE {i}
Title: {metadata.get("source", "Unknown")}
Version: {metadata.get("version", "Unknown")}
Document Type: {metadata.get("document_type", "Unknown")}
Status: {metadata.get("status", "Unknown")}
Audience: {metadata.get("audience", "Unknown")}

Content:
{document.page_content}
"""
        )

    return "\n".join(results)
