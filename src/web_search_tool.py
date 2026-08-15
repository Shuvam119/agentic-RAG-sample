import os

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool


load_dotenv()


@tool
def search_web(query: str) -> str:
    """Search the public web.

    Use only after the knowledge base returned NO_RELEVANT_DOCUMENTATION, or for current events and general knowledge.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return "Web search is unavailable because TAVILY_API_KEY is not configured."

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        search_depth="basic",
        max_results=3,
        include_answer=False,
    )

    results = response.get("results", [])

    if not results:
        return "No relevant web results were found."

    formatted_results = []

    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = " ".join((result.get("content", "") or "").split())

        if len(content) > 400:
            content = content[:400].rsplit(" ", 1)[0] + " […]"

        formatted_results.append(
            f"""
WEB SOURCE {i}

Title: {title}
URL: {url}

Content:
{content}
"""
        )

    return "\n".join(formatted_results)
