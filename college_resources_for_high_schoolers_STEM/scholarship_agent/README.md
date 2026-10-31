# STEM college and scholarship research agent

Code only: not executed, installed, ingested, queried or evaluated during delivery. API compatibility has been checked against documentation, not a live environment. Your Windows reference app.py and scholarship corpus were not accessible here. This project implements the requested build_agent / select_tools / handle_request pattern independently.

## Setup in VS Code / Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set the Nebius chat and embedding model IDs, plain Nebius `/v1` URL, Pinecone API key and dedicated index name. The corpus paths default to the existing files in `../stem_corpus_output/data/`; set `COLLEGE_PATH` or `SCHOLARSHIP_PATH` only if your files are elsewhere. Never paste Markdown links into `.env`. Existing system environment variables take precedence over `.env`.

Select a Nebius chat model supporting OpenAI-compatible tool calling and a distinct embedding model from your account catalog. No OpenAI API key or default OpenAI embedding model is used. `ChatOpenAI` here is a protocol client explicitly pointed to Nebius. Model access/tool-calling support must be verified when you run locally.

Pinecone is deliberately pinned to the 7.x vector API. This project uses Nebius precomputed vectors with `upsert` / `query`, not `upsert_records`. Use a dedicated vector index, not a Pinecone integrated-embedding or Documents API index. Do not upgrade to SDK 10 without adapting index creation and data-plane methods. Index dimension is measured from the configured embedding model. Mismatches are errors; no existing index is deleted or recreated.

## Local preparation and ingestion

Start the interactive Streamlit chat from this directory after ingestion:

```cmd
streamlit run streamlit_app.py
```

The chat UI uses the same agent and enables all six tools on every turn, without running the tool-loadout selector. It keeps conversation history in Streamlit session memory only. Each question may make paid Nebius and Pinecone requests; Tavily is used only if its existing live-verification guard is enabled. The app reads `.env` from this directory first, then falls back to the parent project `.env`. If `COLLEGE_PATH` and `SCHOLARSHIP_PATH` are unset or blank, it uses the corpus files under `../stem_corpus_output/data/`; configured relative paths are resolved from the parent project directory.

```cmd
python app.py prepare
python app.py ingest --create-index
python app.py ask --question "What scholarships in this corpus are available to incoming engineering freshmen?"
python app.py chat
```

`prepare` reads local files and writes prepared chunks without provider calls. `ingest` calls Nebius embeddings and writes Pinecone vectors; it can create an index only with `--create-index`. Existing-index ingestion can omit that flag. Chat/ask call paid providers. Pinecone may need a brief consistency interval after upserts.

### Refreshing the notebook-compatible backend index

The notebook-compatible `notebook_backend.py` keeps normal retrieval read-only. To explicitly add missing chunks and refresh chunks whose stored text differs from the configured local corpus, run this from the `scholarship_agent` directory after checking `.env` and the target index:

```cmd
python -c "from notebook_backend import ingest_corpus; print(ingest_corpus(confirm=True))"
```

This makes embedding requests and writes to both configured Pinecone namespaces. Unchanged chunks are skipped, and chunks no longer present locally are not deleted. The function refuses to run unless `confirm=True` is supplied.

The Streamlit app makes the local college directory lookup available before ingestion. Scholarship semantic/hybrid retrieval requires a matching Pinecone manifest; if it is missing or stale, run `python app.py ingest` from this directory. This calls Nebius embeddings and writes vectors to the configured Pinecone index. Use `--create-index` only when you have confirmed the configured index does not already exist.

Embedding results are cached locally by model, endpoint and text. Changed corpus versions use different Pinecone namespaces; old namespaces remain billable until you deliberately remove them. State manifests prevent current local BM25 data being fused with a different corpus/model snapshot. The Pinecone index must match embedding dimensions and cosine metric. A different model can require a new index.

## Input schemas

College input accepts your `{"colleges": [...]}` JSON, requiring `unitid` and `college_name`. Scholarship input accepts arrays/JSONL or `scholarships`, `programs`, `program_pages`, `records`, `data` list envelopes. `{"status":"accepted","metadata": {...}}` entries from the corpus builder are unwrapped. Non-accepted status records and `page_type: "not_a_program"` records are skipped and counted. Unrecognized shapes and missing program names on records classified as programs fail with actionable errors. Inspect/adjust src/data.py if your local scholarship_metadata.json uses a different schema.

Raw scholarship metadata is not automatically verified. Dates and amounts must come from source evidence. If `raw_file` points to full extracted Markdown, that file is not automatically loaded in this version: the requested metadata corpus is indexed. Include source descriptions/evidence in metadata for richer retrieval. Expanding to source Markdown requires path validation and a separate ingestion update.

## Six tools

- `find_colleges`: exact/candidate lookup, avoids assuming ambiguous matches.
- `search_scholarships`: semantic, keyword or hybrid evidence; returns source records and citations.
- `evaluate_amount`: preserves stated amount scope, rejects unsupported annual-gap calculations.
- `evaluate_eligibility`: conservative explicit-rule checks, usually needs more information for older metadata.
- `find_upcoming_deadlines`: actual ISO dates in a user-specified range, no year extrapolation.
- `verify_official_page`: optional Tavily extraction using your original ledger.

Select up to TOP_K=5 tools from a separate Pinecone namespace with intent safeguards. If a sixth capability is needed in a complex question, ask a follow-up or explicitly choose a broader selection policy; five tools cannot expose all six simultaneously. The agent loop has a recursion limit. Free-form tool/profile dictionaries are flexible, but production deployment should add typed user-profile validation and audited eligibility rules.

Eligibility rules can optionally be curated as `{"field":"citizenship","operator":"in","values":["US"],"evidence":"exact supporting source quote"}`. They must not be manufactured by the agent. `eligibility_rules_complete=true` should be set only after actual review; raw metadata usually lacks a complete set. GPA comparisons require matching explicit `gpa_scale` on source and profile. Results are potentially eligible/ineligible/needs_more_information, never official certification.

An award amount is not assumed annual, USD, renewable or stackable. `award_period="year"` plus `currency="USD"` and numeric award bounds are required for a numerical annual gap. The existing college directory is a discovery input, not proof of STEM programs or university awards.

## Optional live verification

Set ENABLE_LIVE_VERIFICATION=true only when ready. Set CORPUS_SCRIPT and CORPUS_OUTPUT to your existing script and output folder containing ledger.jsonl. The tool refuses a missing ledger rather than creating a fresh spend history. It uses the original retries allocation and conservative reservations, so exhaustion/uncertain requests can stop it. Budget exceptions surface instead of silently spending elsewhere. It makes no student-profile queries and does not ingest refreshed content automatically.

## Fifteen-question evaluation

```cmd
python app.py eval
```

This is an explicit paid operation: 15 agent runs plus retrieval queries. Review eval/cases.json first. Outputs are `.agent_state/evaluation/results.json` and report.md, including semantic versus hybrid top-five matches, selected tools, tool outputs, answers and failures.

No stress-test success/failure findings are claimed before execution. eval/RESULTS_NOT_RUN.md contains the review status. Gold record IDs depend on your actual corpus and must be assigned from source review. Add `gold_record_ids` to cases to compute per-case retrieval hit@5. Faithfulness and abstention correctness require reviewed labels; the harness leaves them null rather than substituting lexical heuristics. Application-cycle coverage, alias resolution, retrieval ranking and tool selection should be examined separately.

For session follow-ups the CLI retains messages in memory, without persisting a student profile. Evaluation traces contain questions and tool outputs: avoid real sensitive student details and manage these local artifacts appropriately.

See ARCHITECTURE.md for the workflow diagram and VALIDATION.md for the execution boundary.

## Documentation consulted

- https://docs.langchain.com/oss/python/langchain/agents
- https://reference.langchain.com/python/langchain/agents/factory/create_agent
- https://reference.langchain.com/python/langchain_openai/chat_models/base/ChatOpenAI
- https://docs.langchain.com/oss/python/integrations/embeddings/nebius
- https://pinecone-io.github.io/python-sdk/pinecone/index.html
- https://sdk.pinecone.io/python/docs/migration/v10-migration.html
