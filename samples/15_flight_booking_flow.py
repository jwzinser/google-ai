"""A FLOW agent for flight booking — contrast this with
14_flight_booking_agent.py. Same book_flight tool, same required fields, but
here WE fix the sequence of steps in code (extract -> validate -> book ->
confirm) instead of letting the model decide turn-by-turn whether to ask a
question or call the tool. Each step is either a narrow, single-purpose model
call or plain Python — never both mixed into one open-ended decision.
"""

from __future__ import annotations

import datetime
from typing import Literal

from pydantic import BaseModel

from common import DEFAULT_MODEL, get_client
from google.genai import types

# --- the tool (same shape as 14_flight_booking_agent.py's book_flight) -----


def book_flight(
    origin: str,
    destination: str,
    departure_date: str,
    passenger_name: str,
    return_date: str | None = None,
    passenger_count: int = 1,
    cabin_class: Literal["economy", "premium_economy", "business", "first"] = "economy",
) -> dict:
    """Book a flight. Called directly by the flow's book() step once
    validate() has confirmed every required field is present — no model
    involved in this step at all."""
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


# --- step 1 (model call): extract whatever details are present -------------


class ExtractedBookingParams(BaseModel):
    origin: str | None = None
    destination: str | None = None
    departure_date: str | None = None
    return_date: str | None = None
    passenger_name: str | None = None
    passenger_count: int | None = None
    cabin_class: str | None = None


def extract_params(user_text: str) -> ExtractedBookingParams:
    """Pull booking details out of free text into a typed object. A missing
    detail stays None — this step only extracts, it never guesses."""
    client = get_client()
    today = datetime.date.today().isoformat()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=f"Today's date is {today}. Extract flight booking details from: {user_text!r}",
        config=types.GenerateContentConfig(
            system_instruction=(
                "Extract only what's explicitly stated or clearly implied "
                "(e.g. 'next Friday' relative to the given today's date). "
                "Leave a field null if it isn't mentioned — never invent one."
            ),
            response_mime_type="application/json",
            response_schema=ExtractedBookingParams,
        ),
    )
    return response.parsed


# --- step 2 (plain Python, no model call): which fields are missing --------

REQUIRED_FIELDS = ["origin", "destination", "departure_date", "passenger_name"]


def validate(params: ExtractedBookingParams) -> list[str]:
    return [f for f in REQUIRED_FIELDS if getattr(params, f) is None]


# --- step 3 (plain Python, no model call): call the tool directly ----------


def book(params: ExtractedBookingParams) -> dict:
    return book_flight(
        origin=params.origin,
        destination=params.destination,
        departure_date=params.departure_date,
        passenger_name=params.passenger_name,
        return_date=params.return_date,
        passenger_count=params.passenger_count or 1,
        cabin_class=params.cabin_class or "economy",
    )


# --- step 4 (model call): phrase the result, nothing more -------------------


def confirm(booking: dict) -> str:
    client = get_client()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=f"Write a short, friendly booking confirmation from this data: {booking}",
    )
    return response.text


# --- the flow itself: the sequence lives here, in code, not in the model ---


def run_flow(user_text: str) -> str:
    params = extract_params(user_text)
    missing = validate(params)
    if missing:
        # A fixed, code-written response — not a model call. The model never
        # gets a chance to decide how (or whether) to ask; we already decided.
        return f"I still need: {', '.join(missing)}."
    booking = book(params)
    return confirm(booking)


if __name__ == "__main__":
    print("--- under-specified request: the FLOW itself blocks on missing fields ---")
    print(run_flow("I need to book a flight to New York next month."))

    print("\n--- fully specified request: extract -> validate -> book -> confirm ---")
    print(run_flow("Book a one-way flight from SFO to JFK on 2026-12-10 for passenger Jane Doe."))
