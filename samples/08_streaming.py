"""Streaming responses token-by-token — what powers a responsive chat UI."""

from common import DEFAULT_MODEL, get_client


def main():
    client = get_client()

    for chunk in client.models.generate_content_stream(
        model=DEFAULT_MODEL,
        contents="Write a 6-line haiku sequence about distributed systems failing gracefully.",
    ):
        print(chunk.text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
