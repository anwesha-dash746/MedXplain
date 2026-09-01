/** Types mirroring backend/app/schemas.py. Keep the two in step. */

export type ExplanationMethod =
  | "attention_rollout"
  | "attention_raw"
  | "attention_grad"
  | "shap"
  | "occlusion"
  | "integrated_gradients";

export interface WordScore {
  text: string;
  score: number;
  start: number;
  end: number;
}

export interface ExplanationPayload {
  method: string;
  description: string;
  /** When false, negative scores carry no "protective" meaning. */
  signed: boolean;
  words: WordScore[];
  top_positive: WordScore[];
  top_negative: WordScore[];
  n_model_evaluations: number;
  compute_ms: number;
  truncated: boolean;
  meta: Record<string, unknown>;
}

export interface PredictResponse {
  note_id: string;
  model: string;
  model_display_name: string;
  outcome: string;
  probability: number;
  predicted_label: number;
  decision_threshold: number;
  risk_band: "low" | "borderline" | "elevated" | "high";
  calibration_note: string;
  explanations: ExplanationPayload[];
  n_chars: number;
  n_tokens: number;
  truncated: boolean;
  inference_ms: number;
  warnings: string[];
}

export interface ModelInfo {
  key: string;
  display_name: string;
  hf_id: string;
  available: boolean;
  outcome?: string | null;
  test_auroc?: number | null;
  test_auprc?: number | null;
  threshold?: number | null;
  max_length?: number | null;
  seed?: number | null;
  note?: string | null;
}

export interface MethodInfo {
  key: ExplanationMethod;
  display_name: string;
  description: string;
  signed: boolean;
  cost: "fast" | "moderate" | "slow";
  typical_ms?: number | null;
}

export interface ExampleNote {
  note_id: string;
  title: string;
  text: string;
  true_label: number;
  outcome: string;
}

export interface HistoryEntry {
  note_id: string;
  timestamp: number;
  model: string;
  method: string;
  probability: number;
  predicted_label: number;
  preview: string;
  n_chars: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  device: string;
  models_loaded: string[];
  models_available: string[];
  default_model: string;
  default_method: string;
}

export interface SeedAggregate {
  mean: number;
  std: number;
  min?: number;
  max?: number;
  n_seeds?: number;
  values?: number[];
}

export interface ResultsPayload {
  available: string[];
  dataset_report?: any;
  baselines?: any;
  transformers?: any;
  evaluation?: any;
  faithfulness?: any;
  stability?: any;
  plausibility?: any;
}
