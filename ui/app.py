import re
import sys
import threading
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

_SRC_AGENT_PY = Path(__file__).resolve().parents[1] / "src" / "agent.py"


def _agent_source_mtime() -> float:
    return _SRC_AGENT_PY.stat().st_mtime_ns


from src.vectorstore import get_stats


st.set_page_config(
    page_title="Documentation Knowledge Agent",
    page_icon="🔎",
    layout="wide",
)


def _warmup():
    """Preload the expensive stacks in the background so the first question
    is fast: the ChromaDB stack (stats + collection) and the
    sentence-transformers/torch embedding model. The agent is created lazily
    on first question, so the graph rebuilds automatically when the source
    changes. The UI has already rendered by the time these finish."""
    try:
        from src.vectorstore import get_collection, get_embeddings, get_stats

        get_collection()
        get_stats()
        get_embeddings()
    except Exception:
        pass


if "warmup_started" not in st.session_state:
    st.session_state.warmup_started = True
    threading.Thread(target=_warmup, daemon=True).start()


st.title("Documentation Knowledge Agent")
st.caption("Answers come from the internal documentation; web search is a fallback.")


def load_agent():
    """Return a cached agent or rebuild when the source changes.

    Streamlit's ``@st.cache_resource`` persists the return value across page
    reruns *and* across server restarts inside the same session, so code
    changes to ``src/agent.py`` are not picked up automatically.  Instead we
    store the agent in ``st.session_state`` and rebuild whenever the source
    file's mtime changes (or the first run of a fresh session).
    """
    mtime = _agent_source_mtime()
    cached = st.session_state.get("_agent")
    if cached and st.session_state.get("_agent_mtime") == mtime:
        return cached

    from src.agent import create_agent

    agent = create_agent()
    st.session_state._agent = agent
    st.session_state._agent_mtime = mtime
    return agent


if "messages" not in st.session_state:
    st.session_state.messages = []


URL_PATTERN = re.compile(r"(?<!\]\()(https?://[^\s\)]+)")

CITATION_PATTERNS = [
    (re.compile(r"【WEB SOURCE (\d+)[^】]*】"), r"[Web source \1]"),
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

    doc_count = sum(1 for s in unique if s["kind"] == "doc")
    web_count = sum(1 for s in unique if s["kind"] == "web")

    with st.expander(f"Sources ({len(unique)}) — {doc_count} doc, {web_count} web"):
        for source in unique:
            if source.get("url"):
                st.markdown(f"- 🌐 [{source['title']}]({source['url']})")
            else:
                st.markdown(f"- 📚 {source['title']}")


def render_source_badges(tools_used):
    parts = []
    if "search_knowledge_base" in tools_used:
        parts.append("📚 Knowledge base")
    if "search_web" in tools_used:
        parts.append("🌐 Web search")
    if parts:
        st.caption("Sources used: " + " · ".join(parts))


# Sidebar
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

        if message["role"] == "assistant" and message.get("tools_used"):
            render_source_badges(message["tools_used"])

        st.markdown(link_urls(message["content"]))

        if message.get("sources"):
            render_sources(message["sources"])

        if message.get("steps"):
            with st.expander("Research steps"):
                st.markdown("\n".join(f"- {step}" for step in message["steps"]))


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
        tools_used = []
        seen = 0

        status_area = st.empty()
        status_area.markdown("🔍 **Thinking…**")

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

                            tools_used.append(name)

                            if name == "search_web":
                                steps.append(
                                    f"🌐 **Web search** for `{query}`"
                                )
                            else:
                                steps.append(
                                    f"📚 **Knowledge base** for `{query}`"
                                )

                    elif msg.type == "tool":
                        content = msg.content or ""
                        sources.extend(extract_sources(content))

                seen = len(messages)
                status_area.markdown(
                    f"🔍 {steps[-1]}…" if steps else "🔍 **Thinking…**"
                )

            answer = (
                messages[-1].content
                if messages
                else "No answer was produced."
            )

        except Exception as exc:
            answer = f"An error occurred while researching: {exc}"
            steps.append(f"⚠️ {exc}")

        status_area.empty()

        render_source_badges(tools_used)

        st.markdown(link_urls(answer))

        if sources:
            render_sources(sources)

        if steps:
            with st.expander("Research steps"):
                st.markdown("\n".join(f"- {step}" for step in steps))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "steps": steps,
            "sources": sources,
            "tools_used": tools_used,
        }
    )
