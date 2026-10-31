# Agentic scholarship Q&A architecture

```mermaid
flowchart TD
    S["Scholarship corpus"] --> L["Schema adapters and provenance"]
    C["College directory"] --> L
    L --> D["LangChain documents and chunks"]
    D --> E["Nebius embeddings and local cache"]
    E --> P["Pinecone document namespace"]
    D --> B["Local BM25 keyword index"]
    T["Six tool descriptions"] --> E
    E --> N["Pinecone tool namespace"]
    Q["Student question and session history"] --> X["Select up to five tools"]
    N --> X
    X --> A["LangChain create_agent with Nebius"]
    A --> R["Scholarship hybrid retrieval"]
    P --> R
    B --> R
    R --> A
    A --> U["College lookup, eligibility, amounts and dates"]
    U --> A
    A --> V["Optional Tavily verification"]
    V --> G["Existing corpus credit ledger and robots policy"]
    G --> A
    A --> F["Cited answer or focused follow-up"]
```

TOP_K=5 selects tools, while RETRIEVAL_K=5 selects evidence chunks. These are independent operations and namespaces.

Hybrid search uses reciprocal-rank fusion of Pinecone cosine dense matches and local BM25 results. Metadata record-type filters keep college directory text out of scholarship evidence. Full program records accompany retrieved chunks so a deadline or eligibility field split across chunks can still be inspected.

Structured college lookup uses exact UNITIDs/names then candidates; it does not infer scholarship links. Scholarship pages stay distinct: grouping pages into one program requires additional reviewed identifiers. No relationship between colleges and scholarships is invented.

Live verification is optional and constrained to known corpus URLs. It invokes the existing stem_corpus.py budget guard with its existing output folder and ledger. It does not bypass the original 900-credit total. Known corpus URLs are not independently certified official sources; review provenance before trusting them.

Agent tools return untrusted evidence. The system prompt requires citations, careful eligibility screening and abstention. Prompt instructions alone cannot guarantee semantic accuracy; use the evaluation harness and human checks before relying on results.
