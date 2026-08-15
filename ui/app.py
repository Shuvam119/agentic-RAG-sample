import re
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))


st.set_page_config(
    page_title="Documentation Knowledge Agent",
    page_icon="🔎",
    layout="wide",
)


st.title("Documentation Knowledge Agent")
st.caption("Agentic RAG-powered documentation assistant")


@st.cache_resource
def load_agent():
    from src.agent import create_agent

    return create_agent()


if "messages" not in st.session_state:
    st.session_state.messages = []


URL_PATTERN = re.compile(r"(?<!\]\()(https?://[^\s\)]+)")

CITATION_PATTERNS = [
    (re.compile(r"【WEB SOURCE (\d+)】"), r"[Web source \1]"),
    (re.compile(r"【(\d+)[^】]*】"), r"[SOURCE \1]"),
]


def link_urls(text):
    text = link_citations(text)
    return URL_PATTERN.sub(lambda m: f"[{m.group(0)}]({m.group(0)})", text)


def link_citations(text):
    for pattern, replacement in CITATION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def extract_sources(content):
    if not isinstance(content, str):
        content = str(content)

    sources = []

    for title, url in re.findall(
        r"WEB SOURCE \d+\s*\n\nTitle:\s*(.*?)\nURL:\s*(\S+)",
        content,
        re.DOTALL,
    ):
        sources.append(
            {"kind": "web", "title": title.strip(), "url": url.strip()}
        )

    for title in re.findall(
        r"SOURCE \d+\nTitle:\s*(.*?)\nVersion:",
        content,
        re.DOTALL,
    ):
        sources.append({"kind": "doc", "title": title.strip()})

    return sources


def render_sources(sources):
    unique = []
    seen = set()

    for source in sources:
        key = source.get("url") or source["title"]
        if key not in seen:
            seen.add(key)
            unique.append(source)

    with st.expander(f"Sources ({len(unique)})"):
        for source in unique:
            if source.get("url"):
                st.markdown(f"- 🌐 [{source['title']}]({source['url']})")
            else:
                st.markdown(f"- 📚 {source['title']}")


# Sidebar
from src.vectorstore import get_stats

with st.sidebar:
    st.header("Knowledge Base")

    stats = get_stats()

    st.write(f"{stats['document_count']} documents indexed")
    st.write(f"{stats['total_chunks']} chunks available")
    st.caption(stats["vectorstore_path"])

    st.divider()

    st.subheader("Products")
    for product in stats["products"]:
        st.write(product)

    st.divider()

    st.subheader("Document types")
    for document_type in stats["document_types"]:
        st.write(f"📄 {document_type}")

    st.divider()

    st.subheader("Document status")
    for status in stats["lifecycle_statuses"]:
        st.write(f"🟢 {status}")

    st.divider()

    st.subheader("Audiences")
    for audience in stats["audiences"]:
        st.write(f"👤 {audience}")

    st.divider()

    st.subheader("Versions")
    for version in stats["versions"]:
        st.write(f"🏷️ {version}")


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message.get("steps"):
            st.caption(
                "\n".join(f"• {step}" for step in message["steps"])
            )

        st.markdown(link_urls(message["content"]))

        if message.get("sources"):
            render_sources(message["sources"])


# Chat input
question = st.chat_input(
    "Ask a question about the documentation..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        agent = load_agent()

        steps = []
        sources = []
        messages = []
        seen = 0

        status_area = st.empty()

        try:

            for state in agent.stream(
                {"messages": st.session_state.messages},
                stream_mode="values",
            ):

                messages = state.get("messages", [])

                for msg in messages[seen:]:

                    if getattr(msg, "tool_calls", None):

                        for call in msg.tool_calls:
                            name = call["name"]
                            query = (call.get("args") or {}).get("query", "")

                            if name == "search_web":
                                steps.append(
                                    f"🌐 **Searching the web** for `{query}`"
                                )
                            else:
                                steps.append(
                                    f"📚 **Searching the knowledge base** for `{query}`"
                                )

                    elif msg.type == "tool":

                        content = msg.content or ""
                        sources.extend(extract_sources(content))

                        snippet = content.strip().replace("\n", " ")
                        if len(snippet) > 160:
                            snippet = snippet[:160] + "…"

                        steps.append(f"↳ `{snippet}`")

                seen = len(messages)
                status_area.markdown("\n\n".join(steps))

            answer = (
                messages[-1].content
                if messages
                else "No answer was produced."
            )

        except Exception as exc:
            answer = f"An error occurred while researching: {exc}"
            steps.append(f"⚠️ {exc}")

        status_area.empty()

        st.markdown(link_urls(answer))

        if sources:
            render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "steps": steps,
            "sources": sources,
        }
    )
