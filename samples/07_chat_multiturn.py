"""Multi-turn chat: the Chat object keeps conversation history for you,
the way an enterprise assistant UI would."""

from common import DEFAULT_MODEL, get_client


def main():
    client = get_client()
    chat = client.chats.create(model=DEFAULT_MODEL)

    turns = [
        "We're migrating from a REST gateway to gRPC internally. What should our rollout plan look like?",
        "Good. Now, what's the biggest risk specifically for mobile clients?",
        "Summarize this whole conversation in 3 bullet points for a Slack update.",
    ]

    for turn in turns:
        response = chat.send_message(turn)
        print(f"> {turn}\n{response.text}\n{'-' * 60}")

    print(f"\n{len(chat.get_history())} messages in history.")


if __name__ == "__main__":
    main()
