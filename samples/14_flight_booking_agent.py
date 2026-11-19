"""How tool PARAMETERS work: type hints + a docstring become a JSON schema
that constrains what the model is allowed to pass, and a required parameter
with no default forces the model to ask a clarifying question instead of
guessing. Demonstrated with a toy flight-booking tool: origin, destination,
dates, and passenger name.
"""

from typing import Literal

from common import DEFAULT_MODEL, get_client
from google.genai import types


def book_flight(
    origin: str,
    destination: str,
    departure_date: str,
    passenger_name: str,
    return_date: str | None = None,
    passenger_count: int = 1,
    cabin_class: Literal["economy", "premium_economy", "business", "first"] = "economy",
) -> dict:
    """Book a flight. Only call this once every required detail is known —
    never guess a date, airport code, or passenger name.

    Args:
        origin: departure airport or city, e.g. "SFO" or "San Francisco".
        destination: arrival airport or city, e.g. "JFK" or "New York".
        departure_date: departure date in ISO 8601 format, YYYY-MM-DD.
        passenger_name: full legal name of the passenger on the booking.
        return_date: return date in ISO 8601 format, YYYY-MM-DD. Omit for one-way.
        passenger_count: number of passengers travelling together.
        cabin_class: one of "economy", "premium_economy", "business", "first".
    """
    # The schema guarantees TYPES match (str, int, one of the Literal values).
    # It does NOT guarantee the VALUES make sense — that's still on us.
    if return_date and return_date < departure_date:
        return {"error": f"return_date {return_date} is before departure_date {departure_date}"}

    confirmation_code = f"{origin[:3].upper()}{destination[:3].upper()}{departure_date.replace('-', '')}"
    return {
        "confirmation_code": confirmation_code,
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
        "return_date": return_date,
        "passenger_name": passenger_name,
        "passenger_count": passenger_count,
        "cabin_class": cabin_class,
        "status": "booked",
    }


AGENT_SYSTEM_INSTRUCTION = (
    "You are a flight booking assistant. book_flight requires origin, "
    "destination, departure_date, and passenger_name. If any of those are "
    "missing or ambiguous, ask ONE clarifying question instead of calling "
    "the tool with a guessed value. Convert relative dates (e.g. 'next "
    "Friday') to YYYY-MM-DD yourself before calling the tool; today's date "
    "will be given to you in the user's message if needed."
)


def new_chat():
    client = get_client()
    return client.chats.create(
        model=DEFAULT_MODEL,
        config=types.GenerateContentConfig(
            tools=[book_flight],
            system_instruction=AGENT_SYSTEM_INSTRUCTION,
        ),
    )


if __name__ == "__main__":
    print("--- fully specified request: the tool call happens immediately ---")
    chat = new_chat()
    response = chat.send_message(
        "Book a one-way flight from SFO to JFK on 2026-12-10 for passenger Jane Doe."
    )
    print(response.text)

    print("\n--- under-specified request: the model should ask, not guess ---")
    chat2 = new_chat()
    response2 = chat2.send_message("I need to book a flight to New York next month.")
    print(response2.text)

    print("\n--- follow-up after clarifying: now the model has everything ---")
    response3 = chat2.send_message(
        "From Chicago, departing 2026-12-15, returning 2026-12-20, "
        "passenger name Alex Rivera, business class."
    )
    print(response3.text)
