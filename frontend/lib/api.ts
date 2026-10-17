import type {
  ClinicalAnalysis,
  ExampleNote,
  ExplanationMethod,
  HealthResponse,
  HistoryEntry,
  MethodInfo,
  ModelInfo,
  PredictResponse,
  ResultsPayload,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly hint?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch (cause) {
    // A network-level failure almost always means the API process is not running,
    // which is worth saying plainly rather than surfacing "Failed to fetch".
    throw new ApiError(
      "Could not reach the MedXplain API.",
      0,
      "Start it with: uvicorn backend.app.main:app --port 8000",
    );
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    let hint: string | undefined;
    try {
      const body = await response.json();
      detail = formatDetail(body.detail) ?? detail;
      hint = body.hint;
    } catch {
      /* response had no JSON body */
    }
    if (response.status === 503) {
      hint ??= "No trained model is available yet. Run scripts/02_train_transformers.py.";
    }
    throw new ApiError(detail, response.status, hint);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/** Shared by the Peer Consult client, which adds its own Authorization header. */
export const apiRequest = request;

/** FastAPI validation errors arrive as a list; turn them into one readable sentence. */
function formatDetail(detail: unknown): string | undefined {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d: any) => {
        const field = Array.isArray(d?.loc) ? d.loc[d.loc.length - 1] : "";
        return field ? `${String(field).replace(/_/g, " ")}: ${d?.msg ?? "invalid"}` : d?.msg ?? "invalid";
      })
      .join("; ");
  }
  return undefined;
}

export interface PredictOptions {
  text: string;
  model?: string;
  method?: ExplanationMethod;
  compareMethods?: ExplanationMethod[];
  explain?: boolean;
  topK?: number;
  maxEvals?: number;
}

export function predict(options: PredictOptions): Promise<PredictResponse> {
  return request<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify({
      text: options.text,
      model: options.model,
      method: options.method,
      compare_methods: options.compareMethods,
      explain: options.explain ?? true,
      top_k: options.topK ?? 15,
      max_evals: options.maxEvals,
    }),
  });
}

export const getHealth = () => request<HealthResponse>("/health");
export const getModels = () => request<ModelInfo[]>("/models");
export const getMethods = () => request<MethodInfo[]>("/methods");
export const getExamples = () => request<ExampleNote[]>("/examples");
export const getHistory = () => request<HistoryEntry[]>("/history");
export const getResults = () => request<ResultsPayload>("/results");
export const clearHistory = () => request<{ cleared: boolean }>("/history", { method: "DELETE" });

export const analyzeClinical = (text: string) =>
  request<ClinicalAnalysis>("/clinical/analyze", { method: "POST", body: JSON.stringify({ text }) });
