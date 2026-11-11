"use client";

import { useEffect, useRef, useState } from "react";

import { AnswerText } from "@/components/AnswerText";
import { Logo } from "@/components/Logo";
import { SourceCard } from "@/components/SourceCard";
import {
  askStream,
  fetchStats,
  sendFeedback,
  type AskResult,
  type Source,
  type Stage,
  type Stats,
} from "@/lib/api";

const EXAMPLES = [
  "How often must a bank carry out periodic KYC updation for high risk customers?",
  "What is the cooling-off period for a digital loan?",
  "What were the small account limits under the 2016 KYC direction?",
  "Can a bank outsource its internal audit function?",
];

const STAGES: { key: Stage; label: string }[] = [
  { key: "rewriting", label: "Understanding the question" },
  { key: "retrieving", label: "Searching the directions" },
  { key: "generating", label: "Writing a cited answer" },
];

interface Turn {
  id: string;
  question: string;
  stage: Stage | null;
  sources: Source[];
  result: AskResult | null;
  error: string | null;
  feedback: 1 | -1 | null;
}

const STATUS_COPY: Record<AskResult["status"], { label: string; tone: string }> = {
  answered: { label: "Answered from the documents", tone: "text-[var(--ok)]" },
  not_found: { label: "Not in the documents", tone: "text-[var(--warn)]" },
  failed_validation: { label: "Could not verify citations", tone: "text-[var(--danger)]" },
};

function money(value: number | null): string {
  if (value === null) return "cost n/a";
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(3)}`;
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [asOf, setAsOf] = useState("");
  const [includeWithdrawn, setIncludeWithdrawn] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [active, setActive] = useState<string | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    fetchStats().then(setStats);
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length]);

  const update = (id: string, patch: Partial<Turn>) =>
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));

  async function submit(text: string) {
    const trimmed = text.trim();
    if (trimmed.length < 3 || busy) return;
    const id = crypto.randomUUID();
    const history = turns
      .filter((t) => t.result?.status === "answered")
      .slice(-3)
      .flatMap((t) => [
        { role: "user" as const, content: t.question },
        { role: "assistant" as const, content: t.result!.answer },
      ]);
    setTurns((prev) => [
      ...prev,
      { id, question: trimmed, stage: null, sources: [], result: null, error: null, feedback: null },
    ]);
    setQuestion("");
    setBusy(true);
    abort.current = new AbortController();
    await askStream(
      { question: trimmed, history, as_of: asOf || null, include_withdrawn: includeWithdrawn },
      {
        onStage: (stage) => update(id, { stage }),
        onSources: (sources) => update(id, { sources }),
        onAnswer: (result) => update(id, { result, stage: null }),
        onError: (error) => update(id, { error, stage: null }),
      },
      abort.current.signal,
    );
    setBusy(false);
  }

  function cite(turnId: string, label: string) {
    const key = `${turnId}-${label}`;
    setActive(key);
    document.getElementById(`source-${key}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  async function rate(turn: Turn, rating: 1 | -1) {
    if (!turn.result || turn.feedback) return;
    if (await sendFeedback(turn.result.trace_id, rating)) update(turn.id, { feedback: rating });
  }

  return (
    <div className="mx-auto flex min-h-dvh max-w-5xl flex-col px-4 sm:px-6">
      <header className="flex items-center justify-between gap-4 py-5">
        <div className="flex items-center gap-3">
          <Logo size={36} />
          <div>
            <h1 className="font-serif text-xl leading-none font-semibold tracking-tight">niyam</h1>
            <p className="mt-1 text-xs text-[var(--text-muted)]">Answers from RBI Master Directions, with checked citations</p>
          </div>
        </div>
        {stats && (
          <p className="hidden text-right text-xs text-[var(--text-muted)] sm:block">
            {stats.documents.toLocaleString("en-IN")} directions · {stats.chunks.toLocaleString("en-IN")} passages
            <br />
            {stats.withdrawn} withdrawn tracked
          </p>
        )}
      </header>

      <main className="flex-1 pb-48">
        {turns.length === 0 && (
          <section className="mt-10 sm:mt-16">
            <h2 className="max-w-2xl font-serif text-3xl leading-tight font-semibold sm:text-4xl">
              Ask what the RBI requires. Get the clause it comes from.
            </h2>
            <p className="mt-4 max-w-2xl text-[var(--text-muted)]">
              Every sentence in an answer links to the paragraph that supports it. If the directions do not
              cover your question, Niyam says so instead of guessing. Set a date to see the rules that applied
              then.
            </p>
            <div className="mt-8 grid gap-3 sm:grid-cols-2">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => submit(example)}
                  className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4 text-left text-sm leading-6 transition hover:border-[var(--highlight)]"
                >
                  {example}
                </button>
              ))}
            </div>
          </section>
        )}

        <div className="mt-6 space-y-10">
          {turns.map((turn) => {
            const labels = turn.result?.citations.map((c) => c.label) ?? [];
            const status = turn.result ? STATUS_COPY[turn.result.status] : null;
            return (
              <section key={turn.id} aria-live="polite">
                <p className="font-serif text-lg font-semibold">{turn.question}</p>
                {turn.result?.rewritten_question && (
                  <p className="mt-1 text-xs text-[var(--text-muted)]">Searched as: {turn.result.rewritten_question}</p>
                )}

                {!turn.result && !turn.error && (
                  <ol className="mt-4 space-y-2 text-sm">
                    {STAGES.map(({ key, label }) => {
                      const order = STAGES.findIndex((s) => s.key === turn.stage);
                      const index = STAGES.findIndex((s) => s.key === key);
                      const state = turn.stage === null ? "pending" : index < order ? "done" : index === order ? "now" : "pending";
                      return (
                        <li key={key} className="flex items-center gap-2 text-[var(--text-muted)]">
                          <span
                            className={`h-2 w-2 rounded-full ${
                              state === "now"
                                ? "animate-pulse bg-[var(--highlight)]"
                                : state === "done"
                                  ? "bg-[var(--ok)]"
                                  : "bg-[var(--border)]"
                            }`}
                          />
                          <span className={state === "now" ? "text-[var(--text)]" : ""}>{label}</span>
                          {key === "retrieving" && turn.sources.length > 0 && (
                            <span className="text-xs">({turn.sources.length} passages)</span>
                          )}
                        </li>
                      );
                    })}
                  </ol>
                )}

                {turn.error && (
                  <p className="mt-4 rounded-lg border border-[var(--danger)] p-3 text-sm text-[var(--danger)]">{turn.error}</p>
                )}

                {turn.result && status && (
                  <div className="mt-4 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.9fr)]">
                    <div>
                      <p className={`text-xs font-semibold tracking-wide uppercase ${status.tone}`}>{status.label}</p>
                      <div className="mt-3 rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5 text-[15px]">
                        {turn.result.status === "answered" ? (
                          <AnswerText text={turn.result.answer} labels={labels} onCite={(l) => cite(turn.id, l)} />
                        ) : (
                          <p className="leading-7">{turn.result.answer}</p>
                        )}
                      </div>
                      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-[var(--text-muted)]">
                        <span>{((turn.result.timings_ms.total ?? 0) / 1000).toFixed(1)} s</span>
                        <span>{money(turn.result.cost_usd)}</span>
                        <span>rules as of {turn.result.as_of}</span>
                        {turn.result.status === "answered" && (
                          <span className="ml-auto flex items-center gap-1">
                            <button
                              type="button"
                              onClick={() => rate(turn, 1)}
                              disabled={turn.feedback !== null}
                              className={`rounded-md px-2 py-1 ring-1 ring-[var(--border)] ${turn.feedback === 1 ? "bg-[var(--surface-muted)] text-[var(--ok)]" : ""}`}
                              aria-label="Helpful"
                            >
                              Helpful
                            </button>
                            <button
                              type="button"
                              onClick={() => rate(turn, -1)}
                              disabled={turn.feedback !== null}
                              className={`rounded-md px-2 py-1 ring-1 ring-[var(--border)] ${turn.feedback === -1 ? "bg-[var(--surface-muted)] text-[var(--danger)]" : ""}`}
                              aria-label="Not helpful"
                            >
                              Not helpful
                            </button>
                          </span>
                        )}
                      </div>
                    </div>
                    {turn.result.citations.length > 0 && (
                      <aside className="space-y-3">
                        <p className="text-xs font-semibold tracking-wide text-[var(--text-muted)] uppercase">Sources</p>
                        {turn.result.citations.map((citation) => (
                          <div key={citation.label} id={`source-${turn.id}-${citation.label}`}>
                            <SourceCard citation={citation} active={active === `${turn.id}-${citation.label}`} />
                          </div>
                        ))}
                      </aside>
                    )}
                  </div>
                )}
              </section>
            );
          })}
        </div>
        <div ref={bottom} />
      </main>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          submit(question);
        }}
        className="fixed inset-x-0 bottom-0 border-t border-[var(--border)] bg-[color-mix(in_srgb,var(--bg)_92%,transparent)] backdrop-blur"
      >
        <div className="mx-auto max-w-5xl px-4 py-4 sm:px-6">
          <div className="flex items-end gap-2 rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-2 focus-within:border-[var(--highlight)]">
            <label htmlFor="question" className="sr-only">
              Question
            </label>
            <textarea
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit(question);
                }
              }}
              rows={1}
              maxLength={1000}
              placeholder="Ask about KYC, digital lending, fraud reporting, deposits…"
              className="max-h-40 min-h-11 flex-1 resize-none bg-transparent px-3 py-2.5 text-[15px] outline-none placeholder:text-[var(--text-muted)]"
            />
            <button
              type="submit"
              disabled={busy || question.trim().length < 3}
              className="h-11 rounded-xl bg-[var(--accent)] px-5 text-sm font-semibold text-[var(--accent-contrast)] transition disabled:opacity-40"
            >
              {busy ? "Working…" : "Ask"}
            </button>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-2 px-1 text-xs text-[var(--text-muted)]">
            <label className="flex items-center gap-2">
              Rules as of
              <input
                type="date"
                value={asOf}
                max={new Date().toISOString().slice(0, 10)}
                onChange={(e) => setAsOf(e.target.value)}
                className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-2 py-1 text-[var(--text)]"
              />
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={includeWithdrawn}
                onChange={(e) => setIncludeWithdrawn(e.target.checked)}
                className="accent-[var(--highlight)]"
              />
              Include withdrawn directions
            </label>
            <span className="ml-auto hidden sm:inline">Not legal advice. Check the official text before acting.</span>
          </div>
        </div>
      </form>
    </div>
  );
}
