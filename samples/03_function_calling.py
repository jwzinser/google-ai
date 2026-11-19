"""Function calling / tool use: Gemini decides when to call your Python functions.

This is the building block behind every "agent" — the model emits a structured
function call, your code executes it, and feeds the result back.
"""

from common import DEFAULT_MODEL, get_client
from google.genai import types


def get_order_status(order_id: str) -> dict:
    """Pretend lookup against an internal order system."""
    fake_db = {
        "A100": {"status": "shipped", "eta_days": 2},
        "A101": {"status": "processing", "eta_days": 5},
    }
    return fake_db.get(order_id, {"status": "not_found"})


def get_refund_policy(region: str) -> dict:
    return {"region": region, "window_days": 30, "restocking_fee_pct": 0}


TOOLS = [get_order_status, get_refund_policy]


def main():
    client = get_client()

    chat = client.chats.create(
        model=DEFAULT_MODEL,
        config=types.GenerateContentConfig(
            tools=TOOLS,  # google-genai auto-generates schemas from type hints + docstrings
            system_instruction="You are a customer support assistant. Use tools; don't guess.",
        ),
    )

    # The SDK's automatic function calling handles the call/execute/respond loop for you.
    response = chat.send_message(
        "Order A101 — where is it, and what's the refund window for the EU?"
    )
    print(response.text)


if __name__ == "__main__":
    main()
