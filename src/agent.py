"""Deterministic knowledge-base-first agent.

Routing is decided in code, never left to the LLM's tool choice:

  1. The knowledge base is ALWAYS searched first.
  2. If the KB returns documentation, the answer is built from those
     documents only. Web search is unreachable.
  3. If the KB returns NO_RELEVANT_DOCUMENTATION and the query names an
     indexed product (e.g. StreamCutPro, PolicyHub), the answer is built
     from an explicit "no internal documentation" context. Web search is
     still unreachable - product questions never hit the public web.
  4. Web search runs only when the KB had nothing AND the query is not a
     product question (genuine general knowledge / current events).

The answering LLM is bound to NO tools, so it cannot fetch external sources
on its own. The graph shape is what guarantees the routing, not a prompt.
"""

import os
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.rag_tool import search_knowledge_base
from src.vectorstore import query_mentions_known_product
from src.web_search_tool import search_web


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

NO_RELEVANT_MARKER = "NO_RELEVANT_DOCUMENTATION"

KB_ANSWER_PROMPT = """
You are a documentation assistant for an enterprise knowledge base containing
the company's internal product documentation (products such as StreamCutPro
and PolicyHub).

You have NO tools. Answer the user's question using ONLY the internal
documentation in the SOURCE blocks of the tool message below.

RULES:
- Build your answer from the SOURCE blocks and cite them as [SOURCE N].
  Only cite numbers that actually appear in the tool output.
- Never mention web search, the public web, or any external product.
- The products in the docs are the company's own (fictional) products. Only
  when the user EXPLICITLY says they found / saw a real, online, or third-party
  product may you add ONE short closing note that a similarly named real-world
  product may exist, while clarifying that your answer comes from the internal
  documentation.
- Prefer the newest document version and a Fresh status.
- If the documentation does not contain the answer, say so clearly; never
  invent facts or sources.
- Answer concisely and do not repeat yourself.
"""

NO_DOCS_ANSWER_PROMPT = """
You are a documentation assistant for an enterprise knowledge base.

The internal knowledge base was searched and returned
NO_RELEVANT_DOCUMENTATION for this question.

RULES:
- This question concerns an internal product or topic. Do NOT search the web
  and do NOT cite or mention any external source.
- State clearly that the internal documentation does not currently contain
  information on this topic.
- Never invent facts, policies, or sources.
- Only when the user EXPLICITLY says they found / saw a real, online, or
  third-party product may you add ONE short closing note that a similarly
  named real-world product may exist.
- Answer concisely and do not repeat yourself.
"""

WEB_ANSWER_PROMPT = """
You are a research assistant. You have NO tools. Answer the user's question
using ONLY the WEB SOURCE blocks of the tool message below.

RULES:
- Cite each fact with [Web source N]. Only cite numbers that actually appear
  in the tool output.
- If there are no web sources, say you could not find information; never
  invent sources or facts.
- Answer concisely and do not repeat yourself.
"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _latest_user_question(state: AgentState) -> str:
    """Return the text of the most recent user message."""
    for message in reversed(state["messages"]):
        if message.type == "human":
            return message.content
    return ""


def _last_tool_content(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if message.type == "tool":
            return message.content or ""
    return ""


def knowledge_base_node(state: AgentState) -> dict:
    query = _latest_user_question(state)
    call_id = f"kb_{uuid.uuid4().hex}"
    result = search_knowledge_base.invoke({"query": query})

    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {"query": query},
                        "id": call_id,
                    }
                ],
                id=f"kb_ai_{uuid.uuid4().hex}",
            ),
            ToolMessage(
                content=result,
                tool_call_id=call_id,
                name="search_knowledge_base",
                id=f"kb_tool_{uuid.uuid4().hex}",
            ),
        ]
    }


def web_node(state: AgentState) -> dict:
    query = _latest_user_question(state)
    call_id = f"web_{uuid.uuid4().hex}"
    result = search_web.invoke({"query": query})

    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_web",
                        "args": {"query": query},
                        "id": call_id,
                    }
                ],
                id=f"web_ai_{uuid.uuid4().hex}",
            ),
            ToolMessage(
                content=result,
                tool_call_id=call_id,
                name="search_web",
                id=f"web_tool_{uuid.uuid4().hex}",
            ),
        ]
    }


def route_after_knowledge_base(state: AgentState) -> str:
    """Deterministic routing: product/internal questions can never reach web."""
    content = _last_tool_content(state)

    if NO_RELEVANT_MARKER not in content:
        return "answer_docs"

    query = _latest_user_question(state)

    if query_mentions_known_product(query):
        return "answer_no_docs"

    return "web"


def _make_answer_node(llm, system_prompt: str):
    def answer_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        response = llm.invoke(messages)
        return {"messages": [response]}

    return answer_node


def create_agent(model_name=MODEL_NAME):
    llm = ChatGroq(model=model_name, temperature=0)

    builder = StateGraph(AgentState)

    builder.add_node("knowledge_base", knowledge_base_node)
    builder.add_node("web", web_node)
    builder.add_node("answer_docs", _make_answer_node(llm, KB_ANSWER_PROMPT))
    builder.add_node("answer_no_docs", _make_answer_node(llm, NO_DOCS_ANSWER_PROMPT))
    builder.add_node("answer_web", _make_answer_node(llm, WEB_ANSWER_PROMPT))

    builder.add_edge(START, "knowledge_base")
    builder.add_conditional_edges(
        "knowledge_base",
        route_after_knowledge_base,
        {
            "answer_docs": "answer_docs",
            "answer_no_docs": "answer_no_docs",
            "web": "web",
        },
    )
    builder.add_edge("web", "answer_web")
    builder.add_edge("answer_docs", END)
    builder.add_edge("answer_no_docs", END)
    builder.add_edge("answer_web", END)

    return builder.compile()
