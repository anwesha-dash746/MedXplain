/** Peer Consult API client and types (backend/app/routers/peer.py). */

import { apiRequest } from "./api";

export type VerificationStatus = "unverified" | "pending" | "verified" | "rejected" | "dev_override";

export interface Verification {
  status: VerificationStatus;
  label: string;
  source: string | null;
  checked_at: number | null;
  note: string | null;
  /** True only when confirmed by a licensing authority. */
  authoritative: boolean;
}

export interface ClinicianProfile {
  id: string;
  display_name: string;
  specialty: string;
  subspecialty: string | null;
  qualifications: string | null;
  years_experience: number | null;
  registration_jurisdiction: string | null;
  interests: string[];
  accepting_consults: boolean;
  verification: Verification;
}

export interface SelfProfile extends ClinicianProfile {
  email: string;
  registration_number: string | null;
  can_use_peer_consult: boolean;
}

export interface PeerStatus {
  database: { available: boolean; error?: string; schema_version?: number };
  verification?: { name: string; available: boolean; message: string; dev_override_enabled: boolean };
  encryption?: { at_rest: boolean; key_source: string; note: string };
  attachments?: { supported: boolean; note: string };
  specialties?: string[];
  consult_types?: { key: string; label: string }[];
  priorities?: { key: string; label: string }[];
  disclaimer: string;
}

export type ConsultStatus =
  | "requested"
  | "accepted"
  | "in_discussion"
  | "response_received"
  | "closed"
  | "declined"
  | "cancelled";

export type Box = "sent" | "received" | "active" | "completed";

export interface ConsultSummary {
  id: string;
  title: string;
  specialty: string;
  consult_type: string;
  consult_type_label: string;
  priority: string;
  priority_label: string;
  status: ConsultStatus;
  status_label: string;
  role: "requester" | "consultant" | "open_pool";
  counterpart_name: string | null;
  open_request: boolean;
  created_at: number;
  updated_at: number;
}

export interface Message {
  id: string;
  author_id: string;
  author_name: string | null;
  author_role: "requester" | "consultant";
  kind: "message" | "opinion";
  content: Record<string, string>;
  created_at: number;
}

export interface ConsultDetail extends ConsultSummary {
  requester: { id: string; name: string | null };
  assigned: { id: string; name: string | null } | null;
  subspecialty: string | null;
  summary: string;
  findings: string;
  diagnosis: string | null;
  question: string;
  matching_preferences: string | null;
  decline_reason: string | null;
  closed_at: number | null;
  messages: Message[];
  allowed_actions: string[];
}

export interface HistoryEvent {
  action: string;
  actor_name: string | null;
  from_status: string | null;
  to_status: string | null;
  created_at: number;
}

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface ConsultationInput {
  client_request_id: string;
  title: string;
  specialty: string;
  subspecialty?: string;
  summary: string;
  findings: string;
  diagnosis?: string;
  question: string;
  consult_type: string;
  priority: string;
  preferred_clinician_id?: string;
  matching_preferences?: string;
  deidentified_confirmed: boolean;
  authorized_confirmed: boolean;
}

export interface RegisterInput {
  email: string;
  password: string;
  display_name: string;
  specialty: string;
  subspecialty?: string;
  qualifications?: string;
  years_experience?: number;
  registration_jurisdiction?: string;
  registration_number?: string;
  interests: string[];
  accepting_consults: boolean;
}

// -- session token (per browser tab; cleared on sign-out) --------------------------

const TOKEN_KEY = "medxplain.peer.token";

export function loadToken(): string | null {
  try {
    return sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function saveToken(token: string | null) {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable: the session lasts until reload */
  }
}

function authed<T>(token: string, path: string, init: RequestInit = {}): Promise<T> {
  return apiRequest<T>(`/peer${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
  });
}

const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const peerApi = {
  status: () => apiRequest<PeerStatus>("/peer/status"),
  register: (input: RegisterInput) => apiRequest<SelfProfile>("/peer/auth/register", post(input)),
  login: (email: string, password: string) =>
    apiRequest<{ token: string; expires_at: number; profile: SelfProfile }>("/peer/auth/login", post({ email, password })),
  logout: (token: string) => authed<void>(token, "/auth/logout", { method: "POST" }),
  me: (token: string) => authed<SelfProfile>(token, "/me"),
  requestVerification: (token: string, jurisdiction: string, number: string) =>
    authed<{ result: string; detail: string; profile: SelfProfile }>(
      token,
      "/me/verification",
      post({ registration_jurisdiction: jurisdiction, registration_number: number }),
    ),
  dashboard: (token: string) => authed<{ counts: Record<Box, number> }>(token, "/dashboard"),
  clinicians: (token: string, params: Record<string, string | number | boolean | undefined>) => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "") q.set(k, String(v));
    return authed<Page<ClinicianProfile>>(token, `/clinicians?${q.toString()}`);
  },
  createConsultation: (token: string, input: ConsultationInput) =>
    authed<{ consultation: ConsultDetail; created: boolean }>(token, "/consultations", post(input)),
  consultations: (token: string, box: Box, page = 1) =>
    authed<Page<ConsultSummary>>(token, `/consultations?box=${box}&page=${page}&page_size=10`),
  consultation: (token: string, id: string) => authed<ConsultDetail>(token, `/consultations/${encodeURIComponent(id)}`),
  act: (token: string, id: string, action: string, reason?: string) =>
    authed<ConsultDetail>(token, `/consultations/${encodeURIComponent(id)}/actions`, post({ action, reason })),
  message: (token: string, id: string, body: Record<string, string>) =>
    authed<ConsultDetail>(token, `/consultations/${encodeURIComponent(id)}/messages`, post(body)),
  history: (token: string, id: string) =>
    authed<HistoryEvent[]>(token, `/consultations/${encodeURIComponent(id)}/history`),
};

// -- client-side mirror of the server's identifier screen (advisory only) ----------

const IDENTIFIER_PATTERNS: [string, RegExp][] = [
  ["an email address", /[\w.+-]+@[\w-]+\.[\w.-]+/],
  ["a phone number", /(?<!\d)(?:\+?\d[\d ().-]{8,}\d)(?!\d)/],
  ["a full date", /\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b/],
  ["a record number", /\b(MRN|medical record (number|no)|NHS (number|no)|SSN)\b/i],
  ["a long identifier-like number", /(?<![\d.])\d{7,}(?![\d.])/],
  ["a de-identification placeholder from a source note", /\[\*\*.*?\*\*\]/],
];

export function findIdentifiers(...texts: (string | undefined)[]): string[] {
  const found: string[] = [];
  for (const text of texts) {
    if (!text) continue;
    for (const [label, re] of IDENTIFIER_PATTERNS) if (!found.includes(label) && re.test(text)) found.push(label);
  }
  return found;
}

export function newRequestId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
  }
}
