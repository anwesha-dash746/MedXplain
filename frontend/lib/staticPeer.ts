/**
 * Static peer responses for Peer Consult.
 *
 * Everything here is fixed text chosen by keyword matching: no network call, no
 * model, no external API. The physician profiles are invented for this app and
 * the replies are general prompts, not advice about any patient. The UI keeps
 * one short line saying the replies are automatically generated, so nobody
 * mistakes them for a real review of their case.
 */

export const GENERATED_NOTE =
  "Replies are automatically generated general prompts, not a review of this case by a physician.";
export const MAX_QUESTIONS = 2;

export interface SimulatedPhysician {
  id: number;
  label: string;
  name: string;
  specialty: string;
  years: number;
}

export const PHYSICIANS: readonly SimulatedPhysician[] = [
  { id: 1, label: "Physician 1", name: "Dr. Aisha Rahman", specialty: "Internal Medicine", years: 12 },
  { id: 2, label: "Physician 2", name: "Dr. Daniel Okafor", specialty: "Emergency Medicine", years: 10 },
  { id: 3, label: "Physician 3", name: "Dr. Priya Nair", specialty: "Respiratory Medicine", years: 9 },
  { id: 4, label: "Physician 4", name: "Dr. Michael Brennan", specialty: "Cardiology", years: 14 },
  { id: 5, label: "Physician 5", name: "Dr. Sofia Lindqvist", specialty: "General Medicine", years: 8 },
];

export type ResponseSetKey = "diagnosis" | "discharge";

export interface ResponseSet {
  key: ResponseSetKey;
  title: string;
  /** One response per physician, in PHYSICIANS order. */
  responses: readonly string[];
}

export const RESPONSE_SETS: Record<ResponseSetKey, ResponseSet> = {
  diagnosis: {
    key: "diagnosis",
    title: "Set A: Diagnosis and Clinical Interpretation",
    responses: [
      "Interpret the symptoms alongside the medical history, examination findings and investigations together rather than in isolation. The information available may not be enough to establish a definitive diagnosis.",
      "Review the timeline: when symptoms started, how severe they were, the initial observations, and how the clinical picture changed afterwards. The direction of change is often as informative as any single finding.",
      "If this is a respiratory presentation, correlate oxygen saturation, the respiratory examination, imaging and the response to treatment. Verify any of these that are missing from the note before drawing conclusions.",
      "Consider the documented cardiovascular history and the relevant investigations. Symptoms alone may not distinguish between possible underlying causes.",
      "Organise the clinical timeline, findings, investigations and treatment response into one summary. Additional information may be needed before the picture is complete.",
    ],
  },
  discharge: {
    key: "discharge",
    title: "Set B: Treatment and Discharge Planning",
    responses: [
      "Verify that medication reconciliation is complete, that the response to treatment is documented, and that the final clinical observations support the plan.",
      "Check that indicators of deterioration, return precautions and the discharge status are all clearly documented.",
      "If this is a respiratory presentation, verify oxygen requirements, the latest respiratory observations and inhaler instructions before discharge.",
      "Confirm the relevant medications, any monitoring instructions and the follow-up arrangements.",
      "Review medications, pending investigations, follow-up, warning signs and the documented care plan as one checklist before the patient leaves.",
    ],
  },
};

const DIAGNOSIS_TERMS = [
  /\bdiagnos\w*/g,
  /\bcause[sd]?\b/g,
  /\bsymptoms?\b/g,
  /\binterpret\w*/g,
  /\bdeteriorat\w*/g,
  /\bfindings?\b/g,
  /\bconditions?\b/g,
];

const DISCHARGE_TERMS = [
  /\btreat\w*/g,
  /\bmedications?\b/g,
  /\bdischarg\w*/g,
  /\bfollow[- ]?ups?\b/g,
  /\bmonitor\w*/g,
  /\bprecautions?\b/g,
  /\bprescri\w*/g,
  /\bmanag\w*/g,
  /\bplan\w*/g,
];

function count(text: string, patterns: RegExp[]): number {
  return patterns.reduce((n, re) => n + (text.match(re)?.length ?? 0), 0);
}

/**
 * Choose the response set for a question. The set with more keyword matches
 * wins; a tie (including no matches at all) defaults to Set A.
 */
export function selectResponseSet(question: string): ResponseSetKey {
  const q = question.toLowerCase();
  const diagnosis = count(q, DIAGNOSIS_TERMS);
  const discharge = count(q, DISCHARGE_TERMS);
  return discharge > diagnosis ? "discharge" : "diagnosis";
}

export interface PeerThread {
  id: number;
  question: string;
  note: string;
  set: ResponseSetKey;
  askedAt: number;
}

export type SubmitResult = { ok: true; thread: PeerThread } | { ok: false; error: string };

/** Validate a submission against the inputs and the per-session limit. */
export function buildThread(note: string, question: string, previous: PeerThread[], now = Date.now()): SubmitResult {
  if (previous.length >= MAX_QUESTIONS) {
    return { ok: false, error: `You have used both questions for this session (${MAX_QUESTIONS}/${MAX_QUESTIONS}).` };
  }
  if (!note.trim()) return { ok: false, error: "Add the discharge note before submitting." };
  if (!question.trim()) return { ok: false, error: "Write your clinical question before submitting." };
  return {
    ok: true,
    thread: {
      id: previous.length + 1,
      question: question.trim(),
      note: note.trim(),
      set: selectResponseSet(question),
      askedAt: now,
    },
  };
}
