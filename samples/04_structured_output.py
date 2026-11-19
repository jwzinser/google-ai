"""Structured output: force the model to return JSON matching a schema.

Useful for enterprise pipelines where downstream code parses the response —
no regex-scraping markdown.
"""

from pydantic import BaseModel

from common import DEFAULT_MODEL, get_client
from google.genai import types


class TicketTriage(BaseModel):
    category: str
    severity: int  # 1-5
    needs_human: bool
    summary: str


def main():
    client = get_client()

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=(
            "Triage this support ticket: 'Our checkout API has been returning "
            "500s for the last 20 minutes, we're losing orders.'"
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TicketTriage,
        ),
    )

    ticket: TicketTriage = response.parsed  # already a validated pydantic object
    print(ticket)
    print(ticket.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
