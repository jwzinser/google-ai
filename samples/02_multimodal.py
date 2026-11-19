"""Multimodal input: ask Gemini about an image and a PDF in the same request.

Vertex AI accepts inline bytes or a gs:// URI. This sample shows both paths.
"""

from common import DEFAULT_MODEL, get_client
from google.genai import types


def analyze_local_image(image_path: str):
    client = get_client()
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            "Describe what's in this image and flag anything that looks like PII.",
        ],
    )
    print(response.text)


def analyze_gcs_pdf(gcs_uri: str):
    """gcs_uri like gs://my-bucket/contract.pdf — this is the enterprise path:
    documents already living in a GCS bucket your project has access to."""
    client = get_client()

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=[
            types.Part.from_uri(file_uri=gcs_uri, mime_type="application/pdf"),
            "Summarize the key obligations and dates in this contract.",
        ],
    )
    print(response.text)


if __name__ == "__main__":
    # Swap in a real local path / gs:// URI to run these.
    # analyze_local_image("sample.png")
    # analyze_gcs_pdf("gs://your-bucket/contract.pdf")
    print("Set a real image path or gs:// URI above, then run this file.")
