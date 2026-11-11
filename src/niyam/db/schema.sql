CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     integer PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id              bigserial PRIMARY KEY,
    source_key      text NOT NULL UNIQUE,
    source_url      text NOT NULL,
    pdf_url         text,
    title           text NOT NULL,
    doc_type        text NOT NULL DEFAULT 'master_direction',
    circular_no     text,
    reference_nos   text[] NOT NULL DEFAULT '{}',
    issued_on       date,
    last_amended_on date,
    amended_on      date[] NOT NULL DEFAULT '{}',
    withdrawn_on    date,
    department      text,
    page_count      integer,
    parser          text NOT NULL DEFAULT 'html-v1',
    content_hash    text NOT NULL,
    fetched_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_issued_on_idx ON documents (issued_on);
CREATE INDEX IF NOT EXISTS documents_withdrawn_on_idx ON documents (withdrawn_on);
CREATE INDEX IF NOT EXISTS documents_reference_nos_idx ON documents USING gin (reference_nos);

CREATE TABLE IF NOT EXISTS supersessions (
    old_document_id     bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    new_document_id     bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    evidence_chunk_id   text,
    detected_at         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (old_document_id, new_document_id),
    CHECK (old_document_id <> new_document_id)
);

CREATE INDEX IF NOT EXISTS supersessions_new_idx ON supersessions (new_document_id);

CREATE TABLE IF NOT EXISTS chunk_sets (
    id          serial PRIMARY KEY,
    name        text NOT NULL UNIQUE,
    strategy    text NOT NULL,
    params      jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id              text PRIMARY KEY,
    chunk_set_id    integer NOT NULL REFERENCES chunk_sets(id) ON DELETE CASCADE,
    document_id     bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal         integer NOT NULL,
    heading_path    text[] NOT NULL DEFAULT '{}',
    heading_text    text NOT NULL DEFAULT '',
    clause_ref      text,
    text            text NOT NULL,
    token_count     integer NOT NULL,
    page_from       integer,
    page_to         integer,
    anchor          text,
    has_table       boolean NOT NULL DEFAULT false,
    tsv             tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', heading_text), 'A') ||
        setweight(to_tsvector('english', text), 'B')
    ) STORED
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (chunk_set_id, document_id, ordinal);

CREATE TABLE IF NOT EXISTS embedding_models (
    id          serial PRIMARY KEY,
    name        text NOT NULL UNIQUE,
    dim         integer NOT NULL CHECK (dim > 0 AND dim <= 2000),
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id    text NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model_id        integer NOT NULL REFERENCES embedding_models(id) ON DELETE CASCADE,
    chunk_set_id    integer NOT NULL REFERENCES chunk_sets(id) ON DELETE CASCADE,
    embedding       vector NOT NULL,
    PRIMARY KEY (model_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS queries (
    trace_id        uuid PRIMARY KEY,
    question        text NOT NULL,
    rewritten       text,
    as_of           date,
    status          text NOT NULL,
    answer          jsonb NOT NULL,
    config          jsonb NOT NULL,
    timings_ms      jsonb NOT NULL,
    usage           jsonb NOT NULL,
    cost_usd        numeric(12, 6) NOT NULL DEFAULT 0,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS queries_created_at_idx ON queries (created_at DESC);

CREATE TABLE IF NOT EXISTS feedback (
    id          bigserial PRIMARY KEY,
    trace_id    uuid NOT NULL REFERENCES queries(trace_id) ON DELETE CASCADE,
    rating      smallint NOT NULL CHECK (rating IN (-1, 1)),
    comment     text,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_trace_idx ON feedback (trace_id);

CREATE TABLE IF NOT EXISTS eval_runs (
    id          uuid PRIMARY KEY,
    name        text NOT NULL,
    config      jsonb NOT NULL,
    metrics     jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

INSERT INTO schema_migrations (version) VALUES (1) ON CONFLICT DO NOTHING;
