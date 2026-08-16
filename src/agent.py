import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent as create_langchain_agent

from src.rag_tool import search_knowledge_base
from src.web_search_tool import search_web


load_dotenv()

SYSTEM_PROMPT = """
You are a documentation assistant for an enterprise knowledge base containing
the company's internal product documentation (products such as StreamCutPro
and PolicyHub).

Your ONLY tools are search_knowledge_base (internal docs) and search_web
(public web).

WORKFLOW - follow exactly:
1. ALWAYS call search_knowledge_base first, for every question, no exceptions.
   Never decide to skip it and go straight to the web.
2. If it returns SOURCE blocks, answer from that documentation and cite the
   SOURCE numbers. Do not use the web for these answers.
3. If it returns NO_RELEVANT_DOCUMENTATION, the docs have nothing on this
   topic: do NOT cite any documentation, and answer from general knowledge or
   search_web instead.

RULES:
- The products in the docs are the company's own (fictional) products. Even if
  a question seems to be about a real-world product with the same name, answer
  from the internal documentation. Only when the user clearly refers to a
  real, online, or third-party product may you add ONE short closing note that
  a similarly named real-world product may exist, while clarifying that your
  answer comes from the internal documentation.
- Call each tool once per query; do not loop or repeat searches.
- Never invent sources. Only cite SOURCE or WEB SOURCE numbers that actually
  appeared in tool output.
- Prefer the newest document version and a Fresh status.
- Answer concisely and do not repeat yourself.
"""


MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

# Cap on graph supersteps. A normal answer needs: KB -> answer (3), or
# KB -> web -> answer (5). This stops runaway tool loops from bloating the
# context (important on low-token free-tier models).
MAX_AGENT_STEPS = 7


def create_agent(model_name=MODEL_NAME):
    llm = ChatGroq(model=model_name, temperature=0)

    return create_langchain_agent(
        model=llm,
        tools=[search_knowledge_base, search_web],
        system_prompt=SYSTEM_PROMPT,
        debug=False,
    ).with_config({"recursion_limit": MAX_AGENT_STEPS})
