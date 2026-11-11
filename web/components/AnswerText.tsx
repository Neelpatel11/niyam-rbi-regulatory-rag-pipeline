import { Fragment, type ReactNode } from "react";

const MARKER = /\[(S\d+(?:\s*,\s*S\d+)*)\]/g;

function withCitations(text: string, onCite: (label: string) => void, valid: Set<string>): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(MARKER)) {
    const index = match.index ?? 0;
    if (index > last) out.push(text.slice(last, index));
    const labels = match[1].split(",").map((l) => l.trim());
    out.push(
      <span key={`${index}-${match[0]}`} className="whitespace-nowrap">
        {labels.map((label) =>
          valid.has(label) ? (
            <button
              key={label}
              type="button"
              onClick={() => onCite(label)}
              className="mx-0.5 inline-flex h-5 min-w-6 items-center justify-center rounded-md bg-[var(--surface-muted)] px-1.5 align-text-top text-[11px] font-semibold text-[var(--text)] ring-1 ring-[var(--border)] transition hover:ring-[var(--highlight)]"
              aria-label={`Show source ${label}`}
            >
              {label.slice(1)}
            </button>
          ) : null,
        )}
      </span>,
    );
    last = index + match[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function AnswerText({
  text,
  labels,
  onCite,
}: {
  text: string;
  labels: string[];
  onCite: (label: string) => void;
}) {
  const valid = new Set(labels);
  const blocks = text.split(/\n{2,}/);
  return (
    <div className="space-y-3 leading-7">
      {blocks.map((block, i) => {
        const lines = block.split("\n");
        const isList = lines.every((line) => /^\s*([-*]|\d+\.)\s+/.test(line));
        if (isList) {
          return (
            <ul key={i} className="list-disc space-y-1 pl-5">
              {lines.map((line, j) => (
                <li key={j}>{withCitations(line.replace(/^\s*([-*]|\d+\.)\s+/, ""), onCite, valid)}</li>
              ))}
            </ul>
          );
        }
        return (
          <p key={i}>
            {lines.map((line, j) => (
              <Fragment key={j}>
                {j > 0 && <br />}
                {withCitations(line, onCite, valid)}
              </Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}
