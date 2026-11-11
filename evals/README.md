# evals

| Path | Contents |
|---|---|
| `golden.jsonl` | The hand-checked RBI golden set (created with `niyam eval review`) |
| `smoke.jsonl` | 34 questions over the synthetic fixture corpus, used by CI |
| `configs/ablations.yaml` | One configuration per ablation row |
| `results/` | Run outputs (`*.json` are git-ignored; `latest_*.md` tables are kept) |
| `labels.jsonl` | Blind human labels for judge calibration |
| `judge_calibration.md` | Judge agreement and Cohen's kappa, written by `niyam eval calibrate` |

## Building the golden set

1. Ingest the corpus you want to evaluate on and embed it.
2. `niyam eval draft --n 60 --unanswerable 20` writes candidates to `candidates.jsonl`.
3. `niyam eval review` walks through each candidate. Accept only questions you would ask, whose
   quote really answers them, and whose expected answer you have checked against the official
   page. For unanswerable candidates, search the corpus for their key terms first.
4. Write the remaining categories by hand, especially `temporal` items (set `as_of`) and
   `multi_clause` items that need two paragraphs.
5. `niyam eval validate` shows the composition by category and split. Aim for the shares in
   [docs/evaluation.md](../docs/evaluation.md).

Add questions from real use too: answers rated "not helpful" in the UI are in the `feedback`
table and are the best source of hard cases.

## Running

```bash
niyam eval tune -n hybrid-rerank                                  # dev split
niyam eval run -n dense-fixed -n dense-structure -n hybrid -n hybrid-rerank --split test
niyam eval run --mode full -n hybrid-rerank --split test --concurrency 1
niyam eval label --results results/<full-run>.json --n 50
niyam eval calibrate --results results/<full-run>.json
```
