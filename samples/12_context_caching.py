"""Context caching: pay once to cache a large, reused context (e.g. a big
policy handbook or codebase) and reuse it cheaply across many requests —
an enterprise cost-control lever for high-volume agents."""

from common import DEFAULT_MODEL, get_client
from google.genai import types


def main():
    client = get_client()

    with open("../data/docs/incident_response.txt") as f:
        large_doc = f.read()

    cache = client.caches.create(
        model=DEFAULT_MODEL,
        config=types.CreateCachedContentConfig(
            contents=[large_doc],
            system_instruction="You are an SRE assistant answering from the runbook only.",
            ttl="3600s",
        ),
    )

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents="For a Sev1, how often do we need to post status updates?",
        config=types.GenerateContentConfig(cached_content=cache.name),
    )
    print(response.text)

    client.caches.delete(name=cache.name)


if __name__ == "__main__":
    main()
