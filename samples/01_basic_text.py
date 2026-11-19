"""Basic text generation, with a system instruction and generation config."""

from common import DEFAULT_MODEL, get_client
from google.genai import types


def main():
    client = get_client()

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents="Give me three risks of relying only on in-house LLM evals.",
        config=types.GenerateContentConfig(
            system_instruction="You are a terse, skeptical AI platform architect.",
            temperature=0.4,
            max_output_tokens=512,
        ),
    )

    print(response.text)
    print("\n--- usage ---")
    print(response.usage_metadata)


if __name__ == "__main__":
    main()
