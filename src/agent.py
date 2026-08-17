"""Deterministic knowledge-base-first agent with intelligent web fallback.

Routing is decided in code, never left to the LLM's tool choice:

  1. Greetings ("hi", "hello", ...) are detected and answered directly —
     no tools, no KB, no web.
  2. The knowledge base is ALWAYS searched first for every substantive
     question. No exceptions.
  3. The answering LLM always has search_web as a tool. The system prompt
     instructs it to answer from the KB docs when they are relevant, and
     to call search_web only when the documentation is missing or
     insufficient. The LLM never gets to skip the KB — the KB tool
     result is always injected into the messages before the LLM is invoked.
"""

import os
import re
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.rag_tool import search_knowledge_base
from src.web_search_tool import search_web


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

NO_RELEVANT_MARKER = "NO_RELEVANT_DOCUMENTATION"

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|howdy|yo|sup|good\s*(morning|afternoon|evening)|"
    r"thanks|thank\s*you|cheers|bye|goodbye|see\s*ya|"
    r"what's\s*up|hiya|hola|greetings)\s*[.!?…]*\s*$",
    re.IGNORECASE,
)


# ---------- System prompts ------------------------------------------------

ANSWER_PROMPT = """\
You are a documentation assistant for an enterprise knowledge base containing
the company's internal product documentation (products such as StreamCutPro
and PolicyHub).  You also have a search_web tool for external information.

The internal knowledge base has already been searched. Its results are in the
tool message below.  You must decide:

  ANSWER FROM KB — when the documentation is relevant and answers the question.
  Cite sources as [SOURCE N].

  CALL search_web — when the documentation does NOT answer the question, is
  only tangentially related, or the user is clearly asking for external /
  real-world information (competitor lists, market equivalents, news,
  current events, public facts, …).  After getting web results, cite them
  as [Web source N] and combine with any useful KB context.

RULES:
- You MUST check the KB results first. Never skip or ignore them.
- If the KB results are relevant, prefer them and cite [SOURCE N]. Do NOT
  call search_web to duplicate information already in the KB.
- If the KB results are relevant but the user's question goes beyond them
  (e.g. "what are its market equivalents?" when the KB only describes the
  product), answer what you can from the KB, then call search_web for the
  rest.
- If the KB results are irrelevant (NO_RELEVANT_DOCUMENTATION) or only
  tangentially related, call search_web.
- If the user explicitly asks about a real-world, online, or third-party
  product, call search_web.
- Never invent facts, policies, or sources.
- Prefer the newest document version and a Fresh status.
- Answer concisely and do not repeat yourself.
"""


GREETING_PROMPT = """\
You are a friendly assistant. Reply to the greeting briefly and naturally.
No tools, no search, no citations. One or two sentences.
"""


# ---------- State ----------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ---------- Helpers --------------------------------------------------------

def _latest_user_question(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if message.type == "human":
            return message.content
    return ""


def _is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.match(text.strip()))


def _execute_tool_calls(tool_calls: list) -> list:
    """Run tool calls and return the corresponding ToolMessage list."""
    messages = []
    for call in tool_calls:
        name = call["name"]
        if name == "search_web":
            result = search_web.invoke(call.get("args") or {})
        else:
            result = f"Unknown tool: {name}"
        messages.append(
            ToolMessage(content=result, tool_call_id=call["id"], name=name)
        )
    return messages


# ---------- Node functions -------------------------------------------------

def knowledge_base_node(state: AgentState) -> dict:
    """Search the KB and inject the results into the message list."""
    query = _latest_user_question(state)
    call_id = f"kb_{uuid.uuid4().hex}"
    result = search_knowledge_base.invoke({"query": query})
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_knowledge_base", "args": {"query": query}, "id": call_id}
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


def answer_node(state: AgentState, llm_with_tools, llm_no_tools) -> dict:
    """The LLM sees KB results and can optionally call search_web."""
    messages = [SystemMessage(content=ANSWER_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)

    if not getattr(response, "tool_calls", None):
        return {"messages": [response]}

    # LLM wants to search the web — execute the tools, then let the LLM
    # compose a final answer that cites both KB and web sources.
    tool_msgs = _execute_tool_calls(response.tool_calls)

    # Build a plain-text context string for the synthesis step.
    # Groq rejects ToolMessages when no tools are bound and sometimes
    # generates tool-call tokens even without tools, so we feed everything
    # as a single HumanMessage.
    question = _latest_user_question(state)
    context_parts = []
    for m in state["messages"]:
        if m.type == "tool":
            context_parts.append(m.content)
    for tm in tool_msgs:
        context_parts.append(tm.content)
    context_text = "\n\n".join(context_parts)

    synthesis_msg = HumanMessage(content=(
        f"Original question: {question}\n\n"
        f"Search results:\n\n{context_text}\n\n"
        "Based on the above, provide a single final answer to the original "
        "question. Cite sources as [SOURCE N] or [Web source N]. Do not call "
        "any tools. Just write the answer."
    ))
    synthesis = [SystemMessage(content=ANSWER_PROMPT), synthesis_msg]

    # Groq may still emit tool-call tokens — catch and fall back to plain text.
    try:
        final = llm_no_tools.invoke(synthesis)
    except Exception:
        final = llm_with_tools.invoke(synthesis)

    return {"messages": [response, *tool_msgs, final]}


# ---------- Routers --------------------------------------------------------

def route_from_start(state: AgentState) -> str:
    if _is_greeting(_latest_user_question(state)):
        return "greeting"
    return "knowledge_base"


# ---------- Agent factory --------------------------------------------------

def create_agent(model_name=MODEL_NAME):
    llm_with_tools = ChatGroq(model=model_name, temperature=0).bind_tools([search_web])
    llm_no_tools = ChatGroq(model=model_name, temperature=0)

    builder = StateGraph(AgentState)

    builder.add_node("knowledge_base", knowledge_base_node)
    builder.add_node("answer", lambda s: answer_node(s, llm_with_tools, llm_no_tools))
    builder.add_node(
        "greeting",
        lambda s: {
            "messages": [
                llm_no_tools.invoke(
                    [SystemMessage(content=GREETING_PROMPT), *s["messages"]]
                )
            ]
        },
    )

    builder.add_conditional_edges(
        START,
        route_from_start,
        {"knowledge_base": "knowledge_base", "greeting": "greeting"},
    )
    builder.add_edge("greeting", END)
    builder.add_edge("knowledge_base", "answer")
    builder.add_edge("answer", END)

    return builder.compile()
