"""A minimal embeddings database: chunk documents, embed them with Gemini,
store vectors + text on disk as a .npz, and answer nearest-neighbor queries
with cosine similarity.

This is deliberately dependency-light (just numpy) to show the mechanics.
For production scale, swap this module for a managed vector store —
Vertex AI Vector Search, AlloyDB/Cloud SQL with pgvector, or Chroma/Pinecone —
while keeping the same embed() calls.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from common import EMBEDDING_MODEL, get_client
from google.genai import types

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "index.npz")


@dataclass
class Chunk:
    text: str
    source: str


def chunk_text(text: str, source: str, max_chars: int = 500) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    for p in paragraphs:
        for i in range(0, len(p), max_chars):
            chunks.append(Chunk(text=p[i : i + max_chars], source=source))
    return chunks


def embed_texts(texts: list[str], task_type: str) -> np.ndarray:
    client = get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return np.array([e.values for e in response.embeddings], dtype=np.float32)


def build_index(docs_glob: str = "../data/docs/*.txt") -> None:
    pattern = os.path.join(os.path.dirname(__file__), docs_glob)
    chunks: list[Chunk] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            chunks.extend(chunk_text(f.read(), source=os.path.basename(path)))

    if not chunks:
        raise RuntimeError(f"No documents found at {pattern}")

    vectors = embed_texts([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    np.savez(
        INDEX_PATH,
        vectors=vectors,
        texts=np.array([c.text for c in chunks], dtype=object),
        sources=np.array([c.source for c in chunks], dtype=object),
    )
    print(f"Indexed {len(chunks)} chunks from {len(set(c.source for c in chunks))} docs -> {INDEX_PATH}")


def search(query: str, top_k: int = 3) -> list[tuple[str, str, float]]:
    """Returns [(text, source, score), ...] sorted by relevance."""
    if not os.path.exists(INDEX_PATH):
        build_index()

    data = np.load(INDEX_PATH, allow_pickle=True)
    vectors, texts, sources = data["vectors"], data["texts"], data["sources"]

    query_vec = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]

    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)
    scores = (vectors @ query_vec) / np.where(norms == 0, 1e-8, norms)

    top_idx = np.argsort(-scores)[:top_k]
    return [(str(texts[i]), str(sources[i]), float(scores[i])) for i in top_idx]


if __name__ == "__main__":
    build_index()
    for text, source, score in search("how much PTO carries over?"):
        print(f"[{score:.3f}] ({source}) {text}")
