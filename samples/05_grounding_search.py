"""Grounding: ground answers in Google Search (or a private enterprise data store)
so responses cite real, current sources instead of hallucinating.

Two grounding modes shown:
1. Google Search grounding — public web, with citations.
2. Vertex AI Search grounding — your own enterprise data store (the thing that
   backs Gemini Enterprise's "search across company data" feature).
"""

from common import DEFAULT_MODEL, get_client
from google.genai import types


def ground_with_google_search():
    client = get_client()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents="What were the most notable AI model releases this week?",
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    print(response.text)
    print("\n--- grounding sources ---")
    for chunk in response.candidates[0].grounding_metadata.grounding_chunks or []:
        print("-", chunk.web.title, chunk.web.uri)


def ground_with_enterprise_data_store(data_store_path: str):
    """data_store_path looks like:
    projects/<project>/locations/global/collections/default_collection/dataStores/<id>
    (created via Vertex AI Search / Gemini Enterprise data store connectors —
    SharePoint, Confluence, Jira, GCS, BigQuery, etc.)
    """
    client = get_client()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents="What's our current PTO carryover policy?",
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    retrieval=types.Retrieval(
                        vertex_ai_search=types.VertexAISearch(datastore=data_store_path)
                    )
                )
            ],
        ),
    )
    print(response.text)


if __name__ == "__main__":
    ground_with_google_search()
    # ground_with_enterprise_data_store("projects/.../dataStores/your-datastore-id")
