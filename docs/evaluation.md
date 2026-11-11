# Evaluation

The evaluation answers four questions:

1. Does retrieval find the passages that answer the question? (recall@k, MRR, nDCG)
2. Is the answer supported by what it cites, and is it right? (faithfulness, correctness,
   citation precision)
3. Does the system refuse when the documents do not answer? (refusal precision, recall, F1)
4. What does it cost and how long does it take? (p50/p95 latency, cost and tokens per query)

## The golden set

`evals/golden.jsonl`, one JSON object per line:

```json
{
  "id": "q0042",
  "question": "How often must a bank carry out periodic KYC updation for high risk customers?",
  "category": "single_fact",
  "expected_answer": "At least once every two years.",
  "evidence": [
    {"source_key": "rbi-md-13141", "quote": "at least once every two years for high risk customers"}
  ],
  "as_of": null,
  "notes": ""
}
```

| Category | Target share | What it tests |
|---|---|---|
| `single_fact` | 35% | One clause answers it |
| `multi_clause` | 20% | Needs two or more passages combined |
| `numeric_table` | 15% | Limits, percentages, periods, table cells |
| `temporal` | 10% | Rules in force at a past date (`as_of` set) |
| `unanswerable` | 20% | Plausible questions the corpus does not answer |

Rules for writing items:

- **Quotes are verbatim** from the official page and short (one or two sentences). Relevance is
  measured against quotes, not chunk IDs, so the same set scores every chunking strategy.
- **Unanswerable questions must be checked** against the corpus by searching for their key terms.
  "Sounds unanswerable" is not enough.
- **One fact per expected answer** where possible, so the judge has a clear reference.
- **No paraphrase of the evidence in the question.** Questions should use the words an analyst
  would use, not the words of the clause.

### Building it

```bash
niyam eval draft --n 60 --unanswerable 20     # LLM proposes candidates from sampled chunks
niyam eval review                             # accept / edit / reject each one by hand
niyam eval validate                           # composition by category and split
```

Drafting only saves typing. Every item is read and corrected by a person before it enters the
golden set. Drafted quotes are checked to be verbatim before they are offered for review.

### Dev and test splits

Each item's split is a stable hash of its ID (30% dev, 70% test). Thresholds and other knobs are
tuned on dev; headline numbers are reported on test.

## Metrics

### Retrieval

A retrieved chunk is **relevant to an evidence quote** if it comes from the same document and
contains at least 60% of the quote's tokens as one contiguous run. Tokens are lowercase
alphanumeric runs with accents removed.

| Metric | Definition |
|---|---|
| recall@k | Share of a question's evidence quotes covered by the top k chunks, averaged over answerable questions |
| hit@5 | Share of questions with at least one covered quote in the top 5 |
| MRR | Mean of 1 / rank of the first relevant chunk |
| nDCG@10 | Binary-relevance nDCG; the ideal ranking places min(#quotes, 10) relevant chunks first |

Retrieval mode evaluates the full ranked candidate list (up to 20). Full mode evaluates the
passages actually sent to the model.

### Generation (full mode)

One judge call per answered question returns:

- **claims**: the answer split into atomic claims, each marked supported, partial or unsupported
  *by the cited passages only*. Faithfulness = mean over claims (supported 1, partial 0.5).
- **correctness** against the reference answer: correct 1, partial 0.5, incorrect 0. Refusals
  and validation failures on answerable questions count as 0.
- **passages**: whether each cited passage supports at least one claim. Citation precision =
  supported citations / citations.

### Refusal

Treated as a binary classifier where the positive class is "should refuse" (the
`unanswerable` category). In retrieval mode the prediction is "top score below threshold"; in
full mode it is "status is not `answered`". Reported: precision, recall, F1, and the false
refusal rate on answerable questions.

### Operations

p50 and p95 end-to-end latency, measured sequentially (`--concurrency 1`) so queueing does not
inflate the numbers; tokens and cost per query from provider usage fields and the price table in
`generation/pricing.py` (prices dated in the file). Unknown prices are reported as unknown,
never as zero.

## Checking the judge

An unchecked LLM judge is a guess. Before quoting faithfulness or correctness:

```bash
niyam eval run --mode full -n hybrid-rerank --split test            # produces a results JSON
niyam eval label --results evals/results/<file>.json --n 50          # label blind, by hand
niyam eval calibrate --results evals/results/<file>.json             # agreement + Cohen's kappa
```

`evals/judge_calibration.md` records percentage agreement, Cohen's kappa and the confusion
matrix for correctness and for "all claims supported". Use a judge model different from the
generator (`NIYAM_JUDGE_MODEL`) to limit self-preference.

## Ablations

`evals/configs/ablations.yaml` defines one configuration per row, changing one component at a
time:

| Config | Changes |
|---|---|
| `dense-fixed` | Dense only, fixed 512-token chunks |
| `dense-structure` | Dense only, structure-aware chunks |
| `lexical-structure` | Full-text only |
| `hybrid` | Dense + full-text with RRF |
| `hybrid-rerank` | + MiniLM cross-encoder |
| `hybrid-rerank-bge` | Cross-encoder swapped for bge-reranker-v2-m3 |
| `hybrid-rerank-rewrite` | + query rewriting |
| `hybrid-rerank-bge-base` | Embeddings swapped for bge-base-en-v1.5 |

```bash
niyam eval run -n dense-fixed -n dense-structure -n hybrid -n hybrid-rerank --split test
niyam eval run --mode full -n hybrid-rerank -n hybrid-rerank-rewrite --split test
```

Each run writes a JSON file with the configuration, environment, per-item results, per-category
and per-split breakdowns, plus `evals/results/latest_<mode>.md` with the table. Runs are also
stored in the `eval_runs` table.

## Continuous integration

CI cannot reach the RBI site or an LLM, so it runs a **smoke evaluation** on a synthetic fixture
corpus (`tests/fixtures/corpus`, 6 documents, 34 questions in `evals/smoke.jsonl`) with the real
local embedding and reranking models. The job fails if recall@5 drops below 0.95 for any
configuration or if the reranked configuration's refusal F1 drops below 0.8. The fixture set is
deliberately small and easy: it catches regressions in parsing, chunking, filtering, fusion and
scoring, and says nothing about real-world quality. Headline numbers come only from the RBI
golden set.
