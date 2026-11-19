"""Does client.models.generate_content() mean exactly one LLM call, or can it
hide a loop of multiple model calls + Python executions? Answer: it depends
on what's in `tools`, and whether Automatic Function Calling (AFC) is on.

- No tools, or a server-side tool (Google Search, code execution): exactly
  ONE network round trip. Google's backend does any searching/code-running
  internally, inside that one call; the SDK is a passive pipe.
- A Python callable in `tools=[...]`, with AFC on (the default): the SDK
  itself loops, making MULTIPLE generate_content round trips and calling
  your Python function in between, all inside what looks like one Python
  call to you.
- Same Python tool, AFC explicitly disabled: back to exactly ONE round trip
  — but now you get a raw, unexecuted function-call part back, and have to
  run the loop yourself.

This file demonstrates all three with an instrumented tool function.
"""

import contextlib

import httpx

from common import DEFAULT_MODEL, get_client
from google.genai import types

# A call log so we can SEE, from the outside, how many times the SDK
# actually invoked our Python function — proof of the loop, not just a claim.
call_log: list[str] = []


def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: the city name, e.g. "Boston".
    """
    call_log.append(city)
    return {"city": city, "condition": "sunny", "temp_c": 22}


def describe_history(history: list) -> int:
    """Label each automatic_function_calling_history entry: a real LLM
    generation (role == "model") vs. an input fed to the model (role ==
    "user") — these are NOT the same thing, even though both are entries in
    the same list. Returns the count of actual generations."""
    generations = 0
    for i, turn in enumerate(history, start=1):
        is_generation = turn.role == "model"
        generations += is_generation
        kind = "GENERATION (model output)" if is_generation else "input (fed to the model)"
        for part in turn.parts:
            if part.text:
                detail = f"text: {part.text!r}"
            elif part.function_call:
                detail = f"function_call: {part.function_call.name}({dict(part.function_call.args)})"
            elif part.function_response:
                detail = f"function_response: {part.function_response.name} -> {part.function_response.response}"
            else:
                detail = str(part)
            print(f"  [{i}] {turn.role:<6} {kind:<28} {detail}")
    return generations


def no_tools_single_call():
    """Case A: no tools at all. Always exactly one round trip."""
    client = get_client()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents="In one sentence, what is a vector database?",
    )
    print("text:", response.text)
    print("automatic_function_calling_history:", response.automatic_function_calling_history)


def server_side_tool_single_call():
    """Case B: a server-side tool (Google Search). Still exactly one round
    trip from the client's point of view — Google's backend does the
    searching internally, inside that one call."""
    client = get_client()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents="What's one notable AI news item from this week?",
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
    )
    print("text:", response.text[:200], "...")
    print("automatic_function_calling_history:", response.automatic_function_calling_history)


def python_tool_afc_enabled_loops():
    """Case C: a Python callable tool, AFC ON (the default). One Python-level
    call to generate_content, but the SDK itself makes multiple model calls
    and invokes get_weather() for you in between.

    Shown here for the mechanics, NOT as the recommended pattern: the SDK
    itself warns that AFC via generate_content directly is deprecated, and
    a future major version will support AFC only through Chat.send_message()
    (which is what every other agent sample in this repo already uses —
    03, 11, 13, 14). Prefer chats.create() for any real multi-turn/tool use.
    """
    call_log.clear()
    client = get_client()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents="What's the weather in Boston?",
        config=types.GenerateContentConfig(tools=[get_weather]),
    )
    print("text:", response.text)
    print("our function was called", len(call_log), "time(s); it logged:", call_log)

    history = response.automatic_function_calling_history or []
    print(f"\nautomatic_function_calling_history has {len(history)} entries:")
    generations = describe_history(history)
    print(f"-> {generations} of those are actual model generations; the rest are inputs fed to the model.")


def python_tool_afc_disabled_manual_loop():
    """Case D: same Python tool, AFC explicitly OFF. Now generate_content
    really is just one call — it returns an unexecuted function_call part,
    and we have to run the loop by hand to get a final answer."""
    call_log.clear()
    client = get_client()

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents="What's the weather in Boston?",
        config=types.GenerateContentConfig(
            tools=[get_weather],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    call_part = response.candidates[0].content.parts[0].function_call
    print("model asked to call:", call_part.name, dict(call_part.args))
    print("our function was called", len(call_log), "time(s) so far — the SDK did NOT run it.")

    # Step 2: WE execute the tool ourselves, since the SDK won't.
    result = get_weather(**call_part.args)
    print("manually executed get_weather ->", result)

    # Step 3: WE build the follow-up turn and call generate_content AGAIN,
    # ourselves — this is exactly the round trip AFC was doing invisibly above.
    follow_up_contents = [
        types.Content(role="user", parts=[types.Part.from_text(text="What's the weather in Boston?")]),
        response.candidates[0].content,  # the model's own function_call turn
        types.Content(
            role="user",
            parts=[types.Part.from_function_response(name=call_part.name, response=result)],
        ),
    ]
    response2 = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=follow_up_contents,
        config=types.GenerateContentConfig(tools=[get_weather]),
    )
    print("final text, from OUR second call:", response2.text)
    print("our function was called", len(call_log), "time(s) total — exactly once, by us.")


@contextlib.contextmanager
def count_http_calls():
    """A universal way to see exactly how many real network calls happen,
    regardless of which case is running or whether a response field like
    automatic_function_calling_history exists. google-genai uses httpx
    internally and deliberately suppresses httpx's own DEBUG logging (to
    avoid leaking API keys/payloads) — so this hooks httpx.Client.send
    directly, beneath that suppression, instead of fighting it via logging.

    Limitation: this proves the CLIENT-visible call count precisely. It
    cannot see inside a single HTTP response — so for a server-side tool
    (Case B), it still can't tell us how many times Google's backend
    invoked the model to produce that one response. That stays unobservable
    from here no matter what we hook.
    """
    count = 0
    original_send = httpx.Client.send

    def counting_send(self, request, *args, **kwargs):
        nonlocal count
        count += 1
        print(f"  [HTTP call #{count}] {request.method} {request.url.path}")
        return original_send(self, request, *args, **kwargs)

    httpx.Client.send = counting_send
    try:
        yield lambda: count
    finally:
        httpx.Client.send = original_send


if __name__ == "__main__":
    print("=== Case A: no tools — one call ===")
    with count_http_calls() as get_count:
        no_tools_single_call()
    print("total HTTP calls:", get_count())

    print("\n=== Case B: server-side tool (Google Search) — still one call ===")
    with count_http_calls() as get_count:
        server_side_tool_single_call()
    print("total HTTP calls:", get_count())

    print("\n=== Case C: Python tool, AFC on (default) — the SDK loops for you ===")
    with count_http_calls() as get_count:
        python_tool_afc_enabled_loops()
    print("total HTTP calls:", get_count())

    print("\n=== Case D: Python tool, AFC off — one call, WE run the loop ===")
    with count_http_calls() as get_count:
        python_tool_afc_disabled_manual_loop()
    print("total HTTP calls:", get_count())
