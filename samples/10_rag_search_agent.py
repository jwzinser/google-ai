"""A chat/search agent: answers questions grounded in the embeddings database
built in vector_store.py, with citations — the same pattern Gemini Enterprise
uses for "search across your company's documents," just built from scratch.
"""

from common import DEFAULT_MODEL, get_client
from google.genai import types
from vector_store import search


SYSTEM_INSTRUCTION = (
    "You are an internal company search assistant. Answer ONLY using the "
    "provided context snippets. If the answer isn't in the context, say you "
    "don't know and suggest who to ask. Always cite the source filename(s) "
    "you used in square brackets."
)


def answer(question: str, top_k: int = 3) -> str:
    results = search(question, top_k=top_k)
    context = "\n\n".join(f"[{src}]\n{text}" for text, src, _score in results)

    client = get_client()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=f"Context:\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )
    return response.text


def chat_loop():
    """Multi-turn version: keeps conversation history across questions."""
    client = get_client()
    chat = client.chats.create(
        model=DEFAULT_MODEL,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )

    questions = [
        "How much PTO carries over to next year?",
        "And what's the process for requesting it?",
        "Separately — what's the per-day meal expense limit while traveling?",
    ]
    for q in questions:
        results = search(q, top_k=3)
        context = "\n\n".join(f"[{src}]\n{text}" for text, src, _ in results)
        response = chat.send_message(f"Context:\n{context}\n\nQuestion: {q}")
        print(f"> {q}\n{response.text}\n{'-' * 60}")


if __name__ == "__main__":
    print(answer("How much PTO carries over to next year?"))
    print("\n" + "=" * 60 + "\n")
    chat_loop()
