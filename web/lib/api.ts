export type AnswerStatus = "answered" | "not_found" | "failed_validation";

export interface Citation {
  label: string;
  chunk_id: string;
  title: string;
  circular_no: string | null;
  clause_ref: string | null;
  heading_path: string[];
  url: string;
  issued_on: string | null;
  withdrawn_on: string | null;
  superseded: boolean;
  amended_after_as_of: boolean;
  excerpt: string;
}

export interface Source {
  chunk_id: string;
  title: string;
  circular_no: string | null;
  clause_ref: string | null;
  heading_path: string[];
  url: string;
  issued_on: string | null;
  withdrawn_on: string | null;
  rerank_score: number | null;
  fused_score: number;
}

export interface Usage {
  model: string;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
}

export interface AskResult {
  trace_id: string;
  status: AnswerStatus;
  answer: string;
  citations: Citation[];
  sources: Source[];
  question: string;
  rewritten_question: string | null;
  as_of: string;
  retrieval_score: number | null;
  timings_ms: Record<string, number>;
  usage: Usage[];
  cost_usd: number | null;
  attempts: number;
  trace_url: string | null;
}

export interface Stats {
  documents: number;
  withdrawn: number;
  chunks: number;
  last_ingested: string | null;
  chunk_set: string;
  embedding_model: string;
  reranker: string | null;
  llm: string;
}

export interface AskInput {
  question: string;
  history: { role: "user" | "assistant"; content: string }[];
  as_of: string | null;
  include_withdrawn: boolean;
}

export type Stage = "rewriting" | "retrieving" | "generating";

export interface StreamHandlers {
  onStage: (stage: Stage) => void;
  onSources: (sources: Source[]) => void;
  onAnswer: (result: AskResult) => void;
  onError: (message: string) => void;
}

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

function headers(): HeadersInit {
  const key = process.env.NEXT_PUBLIC_API_KEY;
  return {
    "Content-Type": "application/json",
    ...(key ? { Authorization: `Bearer ${key}` } : {}),
  };
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    /* fall through */
  }
  return `Request failed (${response.status})`;
}

/** Parse a text/event-stream body into (event, data) pairs. */
export async function* readEvents(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<{ event: string; data: string }> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (data.length) yield { event, data: data.join("\n") };
      boundary = buffer.indexOf("\n\n");
    }
  }
}

export async function askStream(input: AskInput, handlers: StreamHandlers, signal?: AbortSignal) {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/v1/ask/stream`, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify(input),
      signal,
    });
  } catch (error) {
    if ((error as Error).name !== "AbortError") handlers.onError("Could not reach the API.");
    return;
  }
  if (!response.ok || !response.body) {
    handlers.onError(await errorMessage(response));
    return;
  }
  for await (const { event, data } of readEvents(response.body)) {
    if (event === "status") handlers.onStage(JSON.parse(data).stage as Stage);
    else if (event === "sources") handlers.onSources(JSON.parse(data) as Source[]);
    else if (event === "answer") handlers.onAnswer(JSON.parse(data) as AskResult);
    else if (event === "error") handlers.onError(JSON.parse(data).detail ?? "Something went wrong.");
  }
}

export async function sendFeedback(traceId: string, rating: 1 | -1): Promise<boolean> {
  const response = await fetch(`${API_URL}/v1/feedback`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ trace_id: traceId, rating }),
  });
  return response.ok;
}

export async function fetchStats(): Promise<Stats | null> {
  try {
    const response = await fetch(`${API_URL}/v1/stats`, { headers: headers() });
    return response.ok ? ((await response.json()) as Stats) : null;
  } catch {
    return null;
  }
}
