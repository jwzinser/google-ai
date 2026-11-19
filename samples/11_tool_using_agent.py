"""A general-purpose agent: a loop that lets Gemini pick from several tools
across turns, including the knowledge-base search agent from sample 10 and the
order-lookup tool from sample 03. This is the "create an agent" building block —
in Gemini Enterprise's no-code builder this same tool-routing logic is what the
UI configures for you; here it's explicit.
"""

from common import PRO_MODEL, get_client
from google.genai import types
from vector_store import search as kb_search


def search_knowledge_base(query: str) -> str:
    """Search internal company policy docs (PTO, expenses, incident response).

    Args:
        query: the user's question, in natural language.
    """
    results = kb_search(query, top_k=3)
    return "\n\n".join(f"[{src}] {text}" for text, src, _score in results)


def get_order_status(order_id: str) -> dict:
    """Look up the shipping status of a customer order by ID."""
    fake_db = {"A100": {"status": "shipped", "eta_days": 2}, "A101": {"status": "processing"}}
    return fake_db.get(order_id, {"status": "not_found"})


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '1500 * 0.085'."""
    allowed = set("0123456789.+-*/() ")
    if not set(expression) <= allowed:
        return "error: unsupported characters"
    return str(eval(expression))  # restricted to digits/operators above


AGENT_TOOLS = [search_knowledge_base, get_order_status, calculator]

AGENT_SYSTEM_INSTRUCTION = (
    "You are an internal ops agent. You can search company policy docs, look "
    "up order status, and do arithmetic. Pick the right tool(s) for each "
    "request, chain multiple tool calls if needed, and give a direct final "
    "answer. Never fabricate information a tool should provide."
)


def run_agent(user_request: str) -> str:
    client = get_client()
    chat = client.chats.create(
        model=PRO_MODEL,
        config=types.GenerateContentConfig(
            tools=AGENT_TOOLS,
            system_instruction=AGENT_SYSTEM_INSTRUCTION,
        ),
    )
    # google-genai's automatic function calling runs the full
    # call -> execute -> feed result back -> repeat loop internally.
    response = chat.send_message(user_request)
    return response.text


if __name__ == "__main__":
    print(run_agent(
        "Order A101 is 20% over our $50 expense cap for a client meal — "
        "is that reimbursable, what's the overage amount, and where's the order?"
    ))
