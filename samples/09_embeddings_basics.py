"""Embeddings: turn text into vectors for semantic similarity, clustering,
or as the retrieval leg of a RAG pipeline."""

import numpy as np

from common import EMBEDDING_MODEL, get_client
from google.genai import types


def embed(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    client = get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return np.array([e.values for e in response.embeddings])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    docs = [
        "Our refund window is 30 days from the delivery date.",
        "The cat sat on the windowsill all afternoon.",
        "Customers can return items within a month of receiving them.",
    ]
    vectors = embed(docs)

    print("Similarity between refund policy sentences (should be high):")
    print(round(cosine_sim(vectors[0], vectors[2]), 4))

    print("Similarity between refund policy and unrelated sentence (should be low):")
    print(round(cosine_sim(vectors[0], vectors[1]), 4))


if __name__ == "__main__":
    main()
