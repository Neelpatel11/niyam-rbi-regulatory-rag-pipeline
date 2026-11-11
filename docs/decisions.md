# Design decisions

Short records of the choices that shape the system, what was considered, and what would change
the decision.

---

### 1. Parse the HTML detail page, not the PDF

**Decision.** RBI publishes each Master Direction as an HTML page and as a PDF. Niyam parses the
HTML. The PDF parser exists for local files.

**Why.** The HTML carries the structure directly: `p.head` headings, `<a id>` anchors for every
section, `<ol>` lists, real tables. Citations can deep-link to the clause on the official page.
The PDF route needs layout heuristics for headings and tables and only gives page numbers.
Also, the PDF host serves a bot-protection challenge to scripted clients. Niyam does not try to
get around it.

**Trade-off.** No page numbers. Citations use section, clause number and anchor instead, which is
how compliance teams refer to these documents anyway.

**Revisit if** the HTML stops being published, or a source is PDF-only (use `niyam ingest local`).

---

### 2. Postgres + pgvector instead of a dedicated vector database

**Decision.** One Postgres holds documents, chunks, vectors, the full-text index, query logs,
feedback and eval runs.

**Why.** At this scale (thousands of documents, tens of thousands of chunks) HNSW in pgvector
answers in a few milliseconds. Metadata filters (issue date, withdrawal) are plain SQL in the
same query. There is one thing to back up, migrate and secure, and Cloud SQL supports the
extension.

**Revisit if** the corpus grows past roughly 10M vectors, or filtered recall degrades. Next steps
in order: tune `m`/`ef_search`, partition by year, quantise, then consider Qdrant or Vespa.

---

### 3. Postgres full-text search for the keyword side, and it is not BM25

**Decision.** Keyword retrieval uses a weighted `tsvector` (heading text weight A, body weight B),
a GIN index, OR semantics, and `ts_rank_cd`.

**Why.** It runs on Cloud SQL and stock Postgres with no extension, and it catches what dense
retrieval misses: reference numbers, section numbers and acronyms.

**Honest caveat.** `ts_rank_cd` is a cover-density ranking. It has no IDF term and no term
frequency saturation, so it is not BM25. ParadeDB's `pg_search` provides true BM25 inside
Postgres but is not available on Cloud SQL. The retriever takes ranked lists, so a BM25 backend
can be added without touching fusion or reranking.

---

### 4. Reciprocal rank fusion, then a cross-encoder

**Decision.** Fuse dense and keyword lists with RRF (`k = 60`), then rerank the top 20 with a
cross-encoder.

**Why.** RRF needs no score normalisation between cosine similarity and `ts_rank_cd`, which live
on unrelated scales. The cross-encoder reads question and passage together and produces a score
that is comparable across questions, which is what the refusal threshold needs.

**Cost.** The reranker is by far the slowest stage on CPU. On real RBI passages (around 450
tokens each) MiniLM-L6 took 3.4 to 8.6 s for 20 to 30 candidates on a laptop CPU, against tens of
milliseconds for both retrievers. Truncating passages barely helped (5.5 s at 800 characters
against 8.6 s at full length), which points to every pair being padded to the model's 512-token
limit. The default is therefore 20 candidates with batch size 64. The real fixes, in order of
effort: a GPU or a CPU-optimised instance, a hosted reranker (Cohere, Voyage) behind the same
`Reranker` interface, or a distilled smaller cross-encoder. Each is an ablation row.

---

### 5. Refuse before generating

**Decision.** If the best reranker score is under a threshold, return "Not found in the
documents" without calling the LLM.

**Why.** It is cheaper, faster, and harder to get wrong than asking the model to decide. The model
still has its own `answerable` flag for cases the threshold lets through.

**How the threshold is set.** `niyam eval tune` picks the cut-off that maximises refusal F1 on the
dev split of the golden set. Results are reported on the test split. When the reranker is off,
the signal falls back to cosine similarity with its own threshold; bge cosine scores sit high
even for unrelated text, so that threshold must be tuned separately.

---

### 6. Validate citations on the server and retry once

**Decision.** The model must return passage IDs. The server rejects an answer that cites an ID
that was not in the context, has no markers, or lists citations inconsistently. It retries once
with the specific problems, then refuses.

**Why.** Prompting alone does not guarantee valid citations. A check in code does. Refusing after
the second failure is the safe default for a compliance tool.

---

### 7. Do not stream unchecked text

**Decision.** The streaming endpoint sends progress events and the retrieved sources while
working, and the answer only after validation.

**Why.** Streaming tokens would show users text whose citations might then be rejected. The
progress events keep the interface responsive without that risk.

---

### 8. Time awareness through metadata, not prompts

**Decision.** Every document carries `issued_on`, `amended_on[]` and `withdrawn_on`. Retrieval
filters to documents in force on the requested date. Repeals are linked in a `supersessions`
table.

**Why.** "What was the rule in 2020?" is a filtering problem. Leaving it to the model invites
answers that mix old and new rules.

**Known limit.** RBI updates Master Directions in place. The text on the site is the latest
consolidated version, so an "as of" query returns the right *documents* but not necessarily the
wording they had on that date. Citations carry an "amended after this date" flag so the user
knows. Archiving each version at ingestion time would close this gap going forward.

---

### 9. Golden evidence as quotes, not chunk IDs

**Decision.** Each golden question stores verbatim evidence quotes and the document key. A
retrieved chunk counts as relevant when it contains at least 60% of a quote as one contiguous
run of tokens.

**Why.** Chunk IDs change with every chunking strategy, which would make a chunking ablation
impossible to score fairly. Quotes do not change. The 60% threshold gives credit to a chunk that
holds most of a quote that straddles a boundary.

---

### 10. Local ONNX models by default

**Decision.** Embeddings (`bge-small-en-v1.5`) and reranking (`ms-marco-MiniLM-L-6-v2`) run
locally through fastembed. OpenAI and Gemini embeddings are options.

**Why.** No per-query embedding cost, no data leaves the service for retrieval, the full
retrieval eval runs in CI without secrets, and the models are baked into the container image so
cold starts do not download anything.

---

### 11. Claude as the default generator, behind a small interface

**Decision.** `LLMClient.structured(system, user, schema)` with Anthropic, OpenAI, Gemini and an
offline extractive implementation. The default is `claude-opus-5-5` at medium effort, with
server-side refusal fallbacks enabled.

**Why.** Every provider now supports schema-constrained output, so the answer contract
(`answerable`, `answer`, `citations`) is identical across providers and the eval compares them
fairly. The extractive client lets tests and CI run the whole pipeline without network access.

---

### 12. Custom eval scripts instead of RAGAS

**Decision.** Retrieval metrics, refusal metrics and an LLM judge are implemented in
`niyam/evals`.

**Why.** Full control over the definitions (quote-based relevance, refusal as a binary
classifier, per-category breakdowns), one judge call per answer instead of several, and a judge
that can be calibrated against human labels with `niyam eval calibrate`. RAGAS-style metrics
(faithfulness, answer correctness) are covered by the judge.

**Watch out for** self-preference when the judge and the generator are the same model. Set
`NIYAM_JUDGE_MODEL` to a different model for headline numbers, and report judge agreement with
human labels next to them.
