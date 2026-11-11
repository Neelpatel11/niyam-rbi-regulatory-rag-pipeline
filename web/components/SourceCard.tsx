import type { Citation } from "@/lib/api";

function formatDate(value: string | null): string | null {
  if (!value) return null;
  return new Date(`${value}T00:00:00`).toLocaleDateString("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

export function StatusPill({ citation }: { citation: Citation }) {
  if (citation.withdrawn_on) {
    return (
      <span className="rounded-full bg-[color-mix(in_srgb,var(--danger)_14%,transparent)] px-2 py-0.5 text-[11px] font-medium text-[var(--danger)]">
        Withdrawn {formatDate(citation.withdrawn_on)}
      </span>
    );
  }
  if (citation.superseded) {
    return (
      <span className="rounded-full bg-[color-mix(in_srgb,var(--warn)_14%,transparent)] px-2 py-0.5 text-[11px] font-medium text-[var(--warn)]">
        Partly repealed
      </span>
    );
  }
  return (
    <span className="rounded-full bg-[color-mix(in_srgb,var(--ok)_14%,transparent)] px-2 py-0.5 text-[11px] font-medium text-[var(--ok)]">
      In force
    </span>
  );
}

export function SourceCard({ citation, active }: { citation: Citation; active: boolean }) {
  const where = citation.clause_ref ? `Para ${citation.clause_ref}` : citation.heading_path.at(-1) ?? "";
  return (
    <article
      className={`scroll-mt-24 rounded-xl border bg-[var(--surface)] p-4 transition ${
        active ? "border-[var(--highlight)] shadow-[0_0_0_3px_color-mix(in_srgb,var(--highlight)_25%,transparent)]" : "border-[var(--border)]"
      }`}
    >
      <div className="flex items-start gap-3">
        <span className="mt-0.5 inline-flex h-6 min-w-6 items-center justify-center rounded-md bg-[var(--accent)] px-1.5 text-xs font-semibold text-[var(--accent-contrast)]">
          {citation.label.slice(1)}
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="font-serif text-[15px] leading-snug font-semibold">{citation.title}</h3>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-[var(--text-muted)]">
            {citation.circular_no && <span className="font-mono">{citation.circular_no}</span>}
            {citation.issued_on && <span>Issued {formatDate(citation.issued_on)}</span>}
            <StatusPill citation={citation} />
          </div>
          {citation.heading_path.length > 0 && (
            <p className="mt-2 text-xs text-[var(--text-muted)]">
              {citation.heading_path.join(" › ")}
              {where && citation.clause_ref ? ` · ${where}` : ""}
            </p>
          )}
          <blockquote className="mt-2 border-l-2 border-[var(--highlight)] pl-3 text-sm leading-6 text-[var(--text)]">
            {citation.excerpt}
          </blockquote>
          {citation.amended_after_as_of && (
            <p className="mt-2 text-xs text-[var(--warn)]">
              This direction was amended after the date you asked about. The text shown is the current
              consolidated version.
            </p>
          )}
          <a
            href={citation.url}
            target="_blank"
            rel="noreferrer"
            className="mt-3 inline-flex text-sm font-medium text-[var(--text)] underline decoration-[var(--highlight)] decoration-2 underline-offset-4 hover:decoration-[var(--text)]"
          >
            Open the official text
          </a>
        </div>
      </div>
    </article>
  );
}
