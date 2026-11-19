"""Shared client setup for all samples.

All samples use the `google-genai` SDK, which talks to either:
- Vertex AI (the enterprise path: project/location, IAM, ADC, VPC-SC, CMEK, data residency), or
- the Gemini Developer API (API key, for quick prototyping).

Enterprise usage means Vertex AI mode. Set these env vars (see ../.env.example):
    GOOGLE_GENAI_USE_VERTEXAI=True
    GOOGLE_CLOUD_PROJECT=your-gcp-project-id
    GOOGLE_CLOUD_LOCATION=us-central1
and authenticate once with: gcloud auth application-default login
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

DEFAULT_MODEL = "gemini-3.8-flash"
PRO_MODEL = "gemini-2.5-pro"
EMBEDDING_MODEL = "text-embedding-005"

_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Returns a cached genai Client. Defaults to Vertex AI (enterprise) mode.

    Cached (rather than built fresh per call) so a helper that returns a
    `Chat` without otherwise keeping `client` in scope can't have its
    underlying connection garbage-collected out from under it.
    """
    global _client
    if _client is not None:
        return _client

    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "True").lower() in ("1", "true", "yes")

    if use_vertex:
        project = os.environ["GOOGLE_CLOUD_PROJECT"]
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        _client = genai.Client(vertexai=True, project=project, location=location)
    else:
        # Fallback: consumer Gemini Developer API with an API key.
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    return _client
