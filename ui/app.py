import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))


st.set_page_config(
    page_title="CloudDesk Knowledge Agent",
    page_icon="🔎",
    layout="wide",
)


st.title("CloudDesk Knowledge Agent")
st.caption("Agentic RAG-powered documentation assistant")


@st.cache_resource
def load_agent():
    from src.agent import create_agent

    return create_agent()


if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar
with st.sidebar:
    st.header("Knowledge Base")

    st.write("8 CloudDesk documents indexed")

    st.divider()

    st.subheader("Document types")

    st.write("📘 User Guide")
    st.write("⚙️ Administrator Guide")
    st.write("🔧 Installation Guide")
    st.write("🛠️ Troubleshooting")
    st.write("💻 API Reference")
    st.write("📝 Release Notes")
    st.write("⚠️ Known Issues")

    st.divider()

    st.subheader("Document status")

    st.write("🟢 Current")
    st.write("⚪ Archived")


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
question = st.chat_input(
    "Ask a question about CloudDesk documentation..."
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

        with st.spinner("Agent is researching the documentation..."):

            agent = load_agent()

            result = agent.invoke(
                {
                    "messages": st.session_state.messages
                }
            )

            answer = result["messages"][-1].content

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
