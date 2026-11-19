"""Code execution: Gemini writes and runs Python in a sandbox to compute an
exact answer instead of guessing arithmetic/statistics from text."""

from common import DEFAULT_MODEL, get_client
from google.genai import types


def main():
    client = get_client()

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=(
            "I have monthly revenue [120, 135, 142, 160, 158, 171]. "
            "Compute month-over-month growth rate and the compound monthly growth rate."
        ),
        config=types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.executable_code:
            print("--- generated code ---")
            print(part.executable_code.code)
        if part.code_execution_result:
            print("--- execution result ---")
            print(part.code_execution_result.output)
        if part.text:
            print("--- model text ---")
            print(part.text)


if __name__ == "__main__":
    main()
