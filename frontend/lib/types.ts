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
  calibration?: CalibrationInfo | null;
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

// -- clinical workspace (backend/app/routers/clinical.py) ---------------------------

export interface CalibrationInfo {
  evaluated: boolean;
  well_calibrated: boolean;
  ece?: number | null;
  brier_skill_score?: number | null;
  seed?: number | null;
  criteria: string;
  summary: string;
}

/** A span quoted verbatim from a note; text === note.slice(start, end). */
export interface Evidence {
  text: string;
  start: number;
  end: number;
  section: string;
  section_label: string;
}

export interface SummaryField {
  key: string;
  label: string;
  status: "documented" | "not_documented";
  value: string | null;
  display: string | null;
  evidence: Evidence[];
  n_evidence: number;
  /** Inferred, not quoted. Always shown with its basis. */
  interpretation: string | null;
  interpretation_basis: string | null;
}

export type ReviewStatus = "documented" | "not_found" | "needs_verification" | "not_applicable";

export interface ReviewItem {
  key: string;
  label: string;
  status: ReviewStatus;
  status_label: string;
  rationale: string;
  evidence: Evidence[];
}

export interface ExtractedValue {
  value: string;
  evidence: Evidence;
}

export interface FollowUpInfo {
  patient_died: boolean;
  arrangements: Evidence[];
  timeframes: ExtractedValue[];
  responsible_services: ExtractedValue[];
  pending_investigations: Evidence[];
  medication_list: Evidence[];
  medication_changes: Evidence[];
  return_precautions: Evidence[];
  flags: string[];
}

export interface CaseLibrarySource {
  name: string;
  synthetic: boolean;
  label: string;
}

export interface SimilarCases {
  available: boolean;
  source: CaseLibrarySource;
  reason?: string | null;
  data_needed?: string;
  library_size?: number;
  excluded_duplicates?: number;
  criteria?: { type: string; label: string; detail: string }[];
  query_concepts?: { key: string; label: string }[];
  patient_age?: number | null;
  patient_sex?: string | null;
  n_eligible?: number;
  n_matched?: number;
  outcomes?: {
    key: string;
    label: string;
    count: number;
    n: number;
    rate: number;
    library_rate: number;
  }[];
  recommendations?: {
    category: string;
    n_cases: number;
    n: number;
    frequency: number;
    examples: { text: string; note_id: string }[];
  }[];
  examples?: {
    note_id: string;
    similarity: number;
    shared_conditions: string[];
    age: number | null;
    sex: string | null;
    split: string | null;
    died_in_hospital: boolean;
    readmitted_30d: boolean | null;
    disposition: string | null;
    text: string;
  }[];
}

export interface ClinicalAnalysis {
  case_id: string;
  analysed_at: number;
  n_chars: number;
  sections_found: { key: string; header: string; start: number; end: number }[];
  case_summary: {
    fields: SummaryField[];
    recognised_concepts: { key: string; label: string; group: string; evidence: Evidence[] }[];
  };
  documentation_review: {
    items: ReviewItem[];
    counts: Record<ReviewStatus, number>;
    disclaimer: string;
  };
  follow_up: FollowUpInfo;
  similar_cases: SimilarCases;
  /** Rule-generated prompts; never historical records or treatment advice. */
  ai_suggestions: {
    kind: "guideline" | "documentation";
    text: string;
    basis: string;
    /** External guidance this suggestion points to (guideline suggestions only). */
    sources?: { title: string; publisher: string; url: string }[];
  }[];
  peer_consult: { available: boolean; url: string | null };
  disclaimer: string;
  compute_ms: number;
}
