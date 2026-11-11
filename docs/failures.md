# What broke, and how it was fixed

A running log of problems found while building Niyam. Each entry says what happened, how it was
noticed, and what changed.

---

### The PDF host answers scripted clients with a bot challenge

**What happened.** The first design parsed the PDF of each direction for page numbers. Fetching
`rbidocs.rbi.org.in` from a script returned an HTML challenge page instead of a PDF, while the
`rbi.org.in` detail pages returned normally.

**Fix.** Switched to parsing the HTML detail page, which turned out to carry better structure
anyway (see decision 1). The crawler does not attempt to get past the challenge. The PDF parser
remains for files a user downloads and ingests locally.

**Lesson.** Probe every source before designing around it.

---

### A withdrawn direction looked like a current one

**What happened.** The 2016 KYC Master Direction is still on the site with an "Updated as on
August 14, 2025" note, but it was withdrawn on 4 December 2025 and replaced by entity-specific
2025 directions. The only signal is a background image named `Withdrawn04122025.jpg`.

**Fix.** The metadata parser reads that marker into `withdrawn_on`, retrieval hides withdrawn
documents unless the question is dated before the withdrawal, and the repeal clauses of the new
directions are linked through the `supersessions` table. The fixture corpus reproduces this case
and three smoke questions depend on it.

---

### Every paragraph was silently dropped

**What happened.** The first HTML parse of a 2025 direction produced six elements. The parser
skipped any `<p>` that contained a nested `<p>` or `<li>`, and in selectolax's Lexbor backend
`node.css(...)` also matches the node itself, so every paragraph "contained" one. The same quirk
made every data table look like a layout table.

**Fix.** A `_descendants` helper that excludes the node itself. A unit test now checks that clause
text, list items and tables all survive parsing.

---

### Clause numbering disappeared from older directions

**What happened.** Older directions number sub-clauses with `<ol type="a">` and `<ol type="i">`,
so the letters exist only in the browser's rendering, not in the text. Chunks read "first item,
second item" with no "(a)", "(b)".

**Fix.** The parser computes the label from the list type, `start` attribute and position, and
prefixes it when the text does not already start with it.

---

### Footnote numbers glued onto clause text

**What happened.** Amendment footnotes are `<sup>` numbers inside the text, producing
"(a) 3 The provisions ..." and confusing clause detection.

**Fix.** Numeric `<sup>` elements are removed before text extraction.

---

### A generated column was rejected

**What happened.** The first schema built the `tsvector` from
`array_to_string(heading_path, ' ')`. Postgres refused because generated columns need immutable
expressions and `array_to_string` is only stable.

**Fix.** A plain `heading_text` column is written at insert time and the generated column uses it.

---

### Quote matching merged words across dashes

**What happened.** The eval tokenizer folded text to ASCII, which deleted the em dash in
"valid—document" and produced the single token "validdocument". RBI titles use en and em dashes
throughout ("Commercial Banks – Know Your Customer"), so evidence quotes could fail to match.

**Fix.** Normalisation now strips accents but treats every non-alphanumeric character as a
separator. A unit test covers the case.

---

### The async database driver failed on Windows

**What happened.** psycopg's async driver does not support the Proactor event loop that Python
uses by default on Windows.

**Fix.** The CLI runs every coroutine on a selector loop, and the test suite sets a selector loop
policy. Linux containers are unaffected.

---

### A FastAPI dependency was read as a query parameter

**What happened.** With postponed annotations, an `Annotated[..., Depends(...)]` alias defined
inside `create_app` could not be resolved, so FastAPI treated the services argument as a required
query parameter and returned 422. An integration test caught it.

**Fix.** The dependency and its alias live at module level.

---

### Dense-only configurations never refused

**What happened.** On the smoke set, every dense-only configuration scored a refusal F1 of 0. bge
embeddings give cosine scores above 0.55 even to unrelated passages, so the default threshold
never triggered.

**Fix.** Separate thresholds for the reranker signal and the cosine fallback, both tuned with
`niyam eval tune`. The cross-encoder score separated answerable from unanswerable questions far
better (refusal F1 0.91 at its default threshold on the smoke set), which is one more reason the
reranker is on by default.

---

### A whole direction parsed to nothing

**What happened.** On the first real crawl, one of five KYC directions (Payments Banks) produced
zero elements. Its outer layout table also holds the table of contents, so the share of `#`
links among its cells crossed the contents-table threshold and the entire page body was
discarded as navigation. Other directions passed only because they had more data-table cells.

**Fix.** A contents table may not contain nested tables or more than a few paragraphs. A
regression test reproduces the layout. The pipeline already refuses to store documents with
fewer than three parsed elements, which is how this surfaced instead of silently indexing an
empty document.

---

### Real repeal clauses do not name what they repeal

**What happened.** On the fixture corpus, the 2025 direction's repeal clause cites the 2016
direction by reference number and the link is found. On the real corpus, the 2025 KYC
directions repeal "the existing Directions, instructions, and guidelines relating to Know Your
Customer ... as communicated vide circular DOR.RRC.REC.302/33-01-010/2025-26". The list of
repealed instructions lives in that separate circular, which is not a Master Direction, so no
link to the withdrawn 2016 direction is created.

**Decision.** No fuzzy title-matching to guess links, because a wrong "replaced by" pointer is
worse than none in a compliance tool. Retrieval correctness does not depend on the link: the
withdrawal date read from the page already removes the 2016 direction from current answers and
brings it back for dated questions. Ingesting the repeal circulars themselves is the proper fix
and is on the roadmap.
