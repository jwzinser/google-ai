"""Doing RAG properly: more than one retrieval method over the same
corpus, fused and reranked, plus actual evaluation — not just "it looked
right when I tried it."

Retrieval methods, on the same 8 chunks from vector_store.py's corpus:
  1. Dense / vector search      — cosine similarity over Gemini embeddings
                                   (what 10_rag_search_agent.py already uses)
  2. Lexical search (BM25)      — exact term matching, from scratch
  3. Hybrid search               — fuse method 1 + 2 via Reciprocal Rank
                                   Fusion, so rank position (not raw score
                                   scale) decides the combined order
  4. LLM reranking               — Gemini itself scores each candidate's
                                   relevance, reordering the top of the list

Evaluation, on a small hand-labeled (query -> correct source doc) set:
  - Retrieval metrics: Recall@k, Precision@k, MRR, NDCG@k — computed from
    scratch, for every method above, so they're directly comparable.
  - Generation metric: "groundedness" — Gemini-as-judge scoring whether a
    generated RAG answer is actually supported by the retrieved context,
    the way RAGAS/TruLens-style eval frameworks do this in practice.
"""

import math
import os
import re
from collections import Counter

import numpy as np
from pydantic import BaseModel

from common import DEFAULT_MODEL, get_client
from google.genai import types
from vector_store import INDEX_PATH, build_index, embed_texts

# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


def load_corpus() -> tuple[list[str], list[str], np.ndarray]:
    if not os.path.exists(INDEX_PATH):
        build_index()
    data = np.load(INDEX_PATH, allow_pickle=True)
    return list(data["texts"]), list(data["sources"]), data["vectors"]


# ---------------------------------------------------------------------------
# method 1: dense / vector search
# ---------------------------------------------------------------------------


def dense_order(query: str, vectors: np.ndarray) -> list[int]:
    query_vec = np.array(embed_texts([query], task_type="RETRIEVAL_QUERY")[0])
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)
    scores = (vectors @ query_vec) / np.where(norms == 0, 1e-8, norms)
    return list(np.argsort(-scores))


# ---------------------------------------------------------------------------
# method 2: lexical search (BM25), from scratch
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Okapi BM25, the standard lexical-match baseline — implemented
    directly from the formula, no search library. Dense embeddings are
    usually benchmarked against (or fused with) exactly this, not a
    strawman: BM25 reliably wins on exact term matches (IDs, numbers,
    acronyms) that embeddings can blur together."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokenized = [tokenize(d) for d in documents]
        self.doc_len = [len(d) for d in self.tokenized]
        self.avg_len = sum(self.doc_len) / len(self.doc_len)
        self.n_docs = len(documents)
        self.df: Counter = Counter()
        for doc in self.tokenized:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1)

    def score(self, query: str, doc_index: int) -> float:
        counts = Counter(self.tokenized[doc_index])
        total = 0.0
        for term in tokenize(query):
            if term not in counts:
                continue
            freq = counts[term]
            numerator = freq * (self.k1 + 1)
            denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc_index] / self.avg_len)
            total += self.idf(term) * numerator / denom
        return total

    def order(self, query: str) -> list[int]:
        scores = [self.score(query, i) for i in range(self.n_docs)]
        return sorted(range(self.n_docs), key=lambda i: -scores[i])


# ---------------------------------------------------------------------------
# method 3: hybrid search via Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """Fuse heterogeneous rankers by RANK POSITION, not raw score — cosine
    similarity (roughly 0-1) and BM25 scores (unbounded, corpus-dependent)
    aren't on comparable scales, so averaging them directly would be
    meaningless. RRF sidesteps that entirely: a document's fused score is
    the sum of 1/(k + rank) across every list it appears in. k=60 is the
    conventional constant from the original RRF literature."""
    fused: Counter = Counter()
    for ranked in ranked_lists:
        for rank, doc_index in enumerate(ranked):
            fused[doc_index] += 1.0 / (k + rank + 1)
    return [doc_index for doc_index, _ in fused.most_common()]


# ---------------------------------------------------------------------------
# method 4: LLM reranking
# ---------------------------------------------------------------------------


class RelevanceScore(BaseModel):
    score: int  # 0 (irrelevant) to 10 (perfectly relevant)


def llm_rerank(query: str, candidate_indices: list[int], texts: list[str]) -> list[int]:
    """A cross-encoder-style rerank step, done with Gemini instead of a
    dedicated reranking model: score each candidate's relevance to the
    query individually (it sees ONLY one passage at a time, unlike the
    retrievers above which compare all candidates at once), then resort."""
    client = get_client()
    scored = []
    for i in candidate_indices:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=f"Query: {query!r}\nPassage: {texts[i]!r}\n\nRate how relevant this passage is to the query.",
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=RelevanceScore
            ),
        )
        scored.append((i, response.parsed.score))
    scored.sort(key=lambda pair: -pair[1])
    return [i for i, _ in scored]


# ---------------------------------------------------------------------------
# retrieval-level eval metrics, from scratch
# ---------------------------------------------------------------------------


def mrr(ranked_sources: list[str], relevant_source: str) -> float:
    """1 / rank of the FIRST relevant result (0 if none in the list). Only
    cares about the single best hit's position — blind to everything
    else, which is right when there's one "best" answer that needs to
    surface early."""
    for rank, src in enumerate(ranked_sources, start=1):
        if src == relevant_source:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_sources: list[str], relevant_source: str, total_relevant: int, k: int) -> float:
    """(relevant chunks found in the top k) / (total relevant chunks that
    exist anywhere in the corpus). Ignores ORDER within the top k entirely
    — only "did we surface it at all" matters here."""
    hits = sum(1 for src in ranked_sources[:k] if src == relevant_source)
    return hits / total_relevant


def precision_at_k(ranked_sources: list[str], relevant_source: str, k: int) -> float:
    """(relevant chunks in the top k) / k — how much of what we actually
    returned was junk. Trades off against recall: a bigger k generally
    raises recall while diluting precision."""
    hits = sum(1 for src in ranked_sources[:k] if src == relevant_source)
    return hits / k


def ndcg_at_k(ranked_sources: list[str], relevant_source: str, total_relevant: int, k: int) -> float:
    """Normalized Discounted Cumulative Gain: like recall, but weighted by
    HOW HIGH UP each relevant chunk is (1/log2(rank+1) — a hit's
    contribution shrinks the further down the list it appears), then
    normalized against the best-possible ordering (IDCG) so it's
    comparable across queries with different numbers of relevant chunks.
    The single metric real search-ranking eval leans on most, because
    it's sensitive to both "did we find it" (like recall) and "how high
    up is it" (which recall ignores)."""
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, src in enumerate(ranked_sources[:k], start=1) if src == relevant_source
    )
    ideal_hits = min(k, total_relevant)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# A tiny hand-labeled eval set: (query, the one doc we know actually answers it).
EVAL_SET = [
    ("How much does the daily meal expense limit cost while traveling?", "expense_policy.txt"),
    ("What's the process for a full outage incident?", "incident_response.txt"),
    ("How many days of PTO do I accrue per month?", "pto_policy.txt"),
    ("Is alcohol reimbursable at a client dinner?", "expense_policy.txt"),
    ("How often do we post status updates during a Sev2 incident?", "incident_response.txt"),
    ("What happens to unused PTO at the end of the year?", "pto_policy.txt"),
]


def evaluate_method(name: str, order_fn, texts: list[str], sources: list[str], k: int = 3) -> dict:
    metrics = {"mrr": [], "recall@k": [], "precision@k": [], "ndcg@k": []}
    for query, relevant_source in EVAL_SET:
        order = order_fn(query)
        ranked_sources = [sources[i] for i in order]
        total_relevant = sources.count(relevant_source)

        metrics["mrr"].append(mrr(ranked_sources, relevant_source))
        metrics["recall@k"].append(recall_at_k(ranked_sources, relevant_source, total_relevant, k))
        metrics["precision@k"].append(precision_at_k(ranked_sources, relevant_source, k))
        metrics["ndcg@k"].append(ndcg_at_k(ranked_sources, relevant_source, total_relevant, k))

    averaged = {metric: sum(values) / len(values) for metric, values in metrics.items()}
    print(f"{name:<12} " + "  ".join(f"{m}={v:.3f}" for m, v in averaged.items()))
    return averaged


# ---------------------------------------------------------------------------
# generation-level eval: groundedness, Gemini-as-judge
# ---------------------------------------------------------------------------


class GroundednessVerdict(BaseModel):
    grounded: bool
    justification: str


def rag_answer(query: str, texts: list[str], sources: list[str], vectors: np.ndarray, top_k: int = 3) -> tuple[str, str]:
    order = dense_order(query, vectors)[:top_k]
    context = "\n\n".join(f"[{sources[i]}] {texts[i]}" for i in order)
    client = get_client()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=f"Context:\n{context}\n\nQuestion: {query}",
        config=types.GenerateContentConfig(
            system_instruction="Answer only using the given context."
        ),
    )
    return response.text, context


def judge_groundedness(query: str, answer: str, context: str) -> GroundednessVerdict:
    """The actual mechanism behind RAGAS/TruLens-style 'faithfulness'
    scoring: ask a (usually stronger) model whether the ANSWER is fully
    supported by the CONTEXT alone, independent of whether the answer
    happens to be true in the real world."""
    client = get_client()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=(
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer: {answer}\n\n"
            "Is every claim in the Answer directly supported by the Context "
            "above? Answer strictly based on the Context, not outside knowledge."
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=GroundednessVerdict
        ),
    )
    return response.parsed


def main():
    texts, sources, vectors = load_corpus()
    bm25 = BM25(texts)

    print(f"corpus: {len(texts)} chunks across {len(set(sources))} documents\n")

    print("=== Retrieval methods: averaged metrics across the eval set (k=3) ===")
    dense_fn = lambda q: dense_order(q, vectors)
    bm25_fn = lambda q: bm25.order(q)
    hybrid_fn = lambda q: reciprocal_rank_fusion([dense_fn(q), bm25_fn(q)])
    rerank_fn = lambda q: llm_rerank(q, dense_fn(q)[:5], texts)

    evaluate_method("dense", dense_fn, texts, sources)
    evaluate_method("bm25", bm25_fn, texts, sources)
    evaluate_method("hybrid", hybrid_fn, texts, sources)
    evaluate_method("llm_rerank", rerank_fn, texts, sources)

    print("\n=== Generation-level eval: groundedness (Gemini-as-judge) ===")
    query = EVAL_SET[0][0]
    answer, context = rag_answer(query, texts, sources, vectors)
    verdict = judge_groundedness(query, answer, context)
    print(f"query: {query}")
    print(f"answer: {answer}")
    print(f"grounded: {verdict.grounded} — {verdict.justification}")


if __name__ == "__main__":
    main()
