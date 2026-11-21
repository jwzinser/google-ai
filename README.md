# Gemini Enterprise (Vertex AI) — Capability Samples

Runnable samples showing what you can build with Gemini via Vertex AI, the
enterprise path (IAM-based auth, project/region control, no API keys,
VPC-SC/CMEK-eligible). All samples use the official `google-genai` SDK.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your GCP project
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com
```

Edit `.env`:
```
GOOGLE_GENAI_USE_VERTEXAI=True
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

Then from `samples/`: `python 01_basic_text.py`, etc.

## Jupyter

Every sample also has a notebook version in `notebooks/` (same number/name,
`.ipynb`). They import the same `samples/common.py` and `samples/vector_store.py`
modules — no code is duplicated — split into a "setup & definitions" cell and
a "run it" cell so you can tweak and re-run just the call. Needs `ipykernel`:

```bash
pip install -r requirements.txt ipykernel
jupyter lab notebooks/
```

## What's in here

| File | Capability |
|---|---|
| `01_basic_text.py` | Plain text generation, system instructions, generation config |
| `02_multimodal.py` | Image + PDF understanding (local bytes or `gs://` URIs) |
| `03_function_calling.py` | Tool use — model calls your Python functions |
| `04_structured_output.py` | Forced JSON output validated against a Pydantic schema |
| `05_grounding_search.py` | Grounding with Google Search, and with a Vertex AI Search enterprise data store |
| `06_code_execution.py` | Model writes + runs Python in a sandbox for exact math |
| `07_chat_multiturn.py` | Stateful multi-turn chat |
| `08_streaming.py` | Token streaming for responsive UIs |
| `09_embeddings_basics.py` | Text embeddings + cosine similarity |
| `vector_store.py` | **Embeddings database**: chunks docs, embeds them, persists vectors to disk, does nearest-neighbor search |
| `10_rag_search_agent.py` | **Chat/search agent**: answers questions grounded in the embeddings DB, with citations |
| `11_tool_using_agent.py` | **General agent**: routes between KB search, an order-lookup tool, and a calculator across turns |
| `12_context_caching.py` | Cache a large reused context (handbook, codebase) to cut cost on repeated calls |
| `resource_library.py` | **Citable resource database**: embeds a curated list of AI learning resources and stores them in **Firestore**, with native server-side vector search |
| `13_resource_search_agent.py` | **Learning-resource search agent**: answers "how do I learn X" questions, citing real titles + URLs from the Firestore resource library |
| `14_flight_booking_agent.py` | **Tool parameters**: how type hints/docstrings become a required-field JSON schema, and how that forces the model to ask instead of guessing |
| `15_flight_booking_flow.py` | **Flow agent**: the same book_flight tool, but with the step sequence (extract → validate → book → confirm) fixed in code instead of decided by the model |
| `16_generate_content_internals.py` | **Is it one LLM call or a loop?**: proves, with an instrumented tool function, when `generate_content` is exactly one round trip vs. when the SDK loops multiple calls for you |
| `17_rag_methods_and_eval.py` | **Doing RAG properly**: dense, BM25 (from scratch), hybrid (RRF), and LLM reranking on the same corpus — plus retrieval metrics (Recall/Precision/MRR/NDCG) and a Gemini-as-judge groundedness eval, all from scratch |

Each `.py` file above has a matching notebook in `notebooks/`.

`notebooks/00_build_vector_database.ipynb` walks through building the embeddings
database step by step (chunk → embed → persist → query) instead of calling
`vector_store.build_index()` as one opaque function — useful if you want to see
or tweak each stage.

## The "create an agent" and "embeddings database" pieces

- `vector_store.py` is a from-scratch embeddings database: `build_index()` chunks
  the `.txt` files in `data/docs/`, embeds them with `text-embedding-005`, and
  saves vectors to `data/index.npz`. `search()` does cosine-similarity retrieval.
  Swap it for Vertex AI Vector Search, pgvector, or Chroma at scale — the
  `embed_texts()` calls stay the same.
- `10_rag_search_agent.py` is a **search/chat agent**: every question is
  embedded, matched against the vector store, and the top chunks are stuffed
  into the prompt as grounded context — the same retrieval-augmented pattern
  behind Gemini Enterprise's "search your company's docs" feature.
- `11_tool_using_agent.py` is a general **agent**: one `chat.send_message()`
  call lets Gemini choose among multiple tools (KB search, order lookup,
  calculator), chain several tool calls, and return one answer. The
  `google-genai` SDK runs the call → execute → feed-result-back loop for you.

## Citable resource database (Firestore)

`data/resources.json` is a curated, verified list of real AI learning resources
— papers, docs, courses — across agents, agent toolkits, LLM fundamentals, RAG,
embeddings, and search. `resource_library.py` embeds each entry and stores it
in a **real Google Cloud database** — Firestore — instead of a local file, so
`13_resource_search_agent.py` can cite sources with real titles and URLs.

**One-time setup**, in addition to the base setup above:

```bash
pip install google-cloud-firestore   # already in requirements.txt

# 1. Enable the Firestore API
gcloud services enable firestore.googleapis.com

# 2. Create a Native-mode Firestore database in your project (skip if one exists).
#    Using a named database (not the project's "(default)" one) is fine — just
#    set GOOGLE_FIRESTORE_DATABASE in .env to match.
gcloud firestore databases create --database=ai-resource --location=us-central1 --type=firestore-native

# 3. Create a vector index on the "embedding" field — required before find_nearest() works
gcloud firestore indexes composite create --database=ai-resource \
  --collection-group=resources --query-scope=COLLECTION \
  --field-config field-path=embedding,vector-config='{"dimension":"768","flat":"{}"}'
```

Then build the index and query it:

```bash
cd samples && python resource_library.py          # or run notebooks/00_build_resource_library.ipynb
python 13_resource_search_agent.py                 # or notebooks/13_resource_search_agent.ipynb
```

Cost: effectively $0 for this dataset size (24 small documents) — Firestore's
free tier covers 1 GiB storage and 50,000 reads / 20,000 writes per day.

## Note on "Gemini Enterprise" the product

This repo targets the **Vertex AI Gemini API** — the programmable layer
enterprises build on. The separate **Gemini Enterprise SaaS product**
(formerly Agentspace) wraps similar primitives in a no-code UI: a company
chat assistant, a drag-and-drop agent builder, prebuilt connectors
(SharePoint, Confluence, Jira, Salesforce, ServiceNow, Workspace), and admin
governance controls. Ask if you'd like a tour doc for that instead.
