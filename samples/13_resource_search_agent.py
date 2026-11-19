"""A learning-resource search agent: answers "how do I learn X" questions by
retrieving real, citable resources from the Firestore-backed resource_library
(see resource_library.py) and always linking its sources.
"""

from common import DEFAULT_MODEL, get_client
from google.genai import types
from resource_library import search as search_resources


def find_learning_resources(query: str, topic: str | None = None) -> str:
    """Search the curated AI learning-resource library (agents, agent toolkits,
    LLM fundamentals, RAG, embeddings, search) and return citable matches.

    Args:
        query: what the user wants to learn, in natural language.
        topic: optional filter — one of "agents", "agent-toolkit",
            "llm-fundamentals", "rag", "embeddings", "search", "enterprise-search".
    """
    hits = search_resources(query, top_k=5, topic=topic)
    if not hits:
        return "No matching resources found."
    return "\n\n".join(f"- {h['title']} ({h['url']}) — {h['summary']}" for h in hits)


AGENT_SYSTEM_INSTRUCTION = (
    "You help people learn AI engineering topics. Always call "
    "find_learning_resources before answering, then recommend 2-4 of the "
    "results. Cite every recommendation as a markdown link: [Title](URL). "
    "Never invent a resource or URL that didn't come from the tool."
)


def ask(question: str) -> str:
    client = get_client()
    chat = client.chats.create(
        model=DEFAULT_MODEL,
        config=types.GenerateContentConfig(
            tools=[find_learning_resources],
            system_instruction=AGENT_SYSTEM_INSTRUCTION,
        ),
    )
    response = chat.send_message(question)
    return response.text


if __name__ == "__main__":
    print(ask("I want to learn how to build a RAG-based search agent. Where should I start?"))
