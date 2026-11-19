"""A citable resource library backed by a real managed database: Firestore.

Each entry in data/resources.json (title, url, topic, summary) is embedded
with Gemini and upserted into a Firestore collection as a document with a
Vector field. Queries embed the question and run Firestore's native
find_nearest() vector search server-side — no local index file, no
in-memory cosine-similarity loop.

One-time setup (see ../README.md for the full walkthrough):
  1. Enable the Firestore API and create a Native-mode database in your project.
  2. Create a vector index on the "embedding" field:
       gcloud firestore indexes composite create \\
         --collection-group=resources --query-scope=COLLECTION \\
         --field-config field-path=embedding,vector-config='{"dimension":"768","flat":"{}"}'
  3. pip install google-cloud-firestore
"""

from __future__ import annotations

import json
import os

from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from common import EMBEDDING_MODEL, get_client

COLLECTION = "resources"
RESOURCES_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "resources.json")


def get_firestore_client() -> firestore.Client:
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    database = os.getenv("GOOGLE_FIRESTORE_DATABASE", "(default)")
    return firestore.Client(project=project, database=database)


def load_resources() -> list[dict]:
    with open(RESOURCES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_texts(texts: list[str], task_type: str):
    client = get_client()
    from google.genai import types

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in response.embeddings]


def build_index() -> None:
    """Embed every resource and upsert it into Firestore."""
    resources = load_resources()
    vectors = embed_texts(
        [f"{r['title']}. {r['summary']}" for r in resources],
        task_type="RETRIEVAL_DOCUMENT",
    )

    db = get_firestore_client()
    batch = db.batch()
    for resource, vector in zip(resources, vectors):
        doc_ref = db.collection(COLLECTION).document(resource["id"])
        batch.set(doc_ref, {**resource, "embedding": Vector(vector)})
    batch.commit()

    print(f"Upserted {len(resources)} resources into Firestore collection '{COLLECTION}'")


def search(query: str, top_k: int = 3, topic: str | None = None) -> list[dict]:
    """Nearest-neighbor search over the resource library, run server-side by Firestore.

    Returns dicts with title/url/topic/type/summary/distance (lower = more relevant).
    """
    query_vector = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]

    db = get_firestore_client()
    collection = db.collection(COLLECTION)
    if topic:
        collection = collection.where("topic", "==", topic)

    results = collection.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_vector),
        distance_measure=DistanceMeasure.COSINE,
        limit=top_k,
        distance_result_field="distance",
    )

    hits = []
    for doc in results.stream():
        data = doc.to_dict()
        data.pop("embedding", None)
        hits.append(data)
    return hits


if __name__ == "__main__":
    build_index()
    for hit in search("how do I learn to build a RAG agent?"):
        print(f"[{hit['distance']:.3f}] {hit['title']} — {hit['url']}")
