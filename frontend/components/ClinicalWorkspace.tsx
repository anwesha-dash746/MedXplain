"use client";

import { useState } from "react";
import { ImportanceChart } from "./ImportanceChart";
import { ResultsView } from "./ResultsView";
import { HowToRead, isCalibrated } from "./PlainLanguage";
import type { ClinicalAnalysis, PredictResponse, SimilarCases } from "@/lib/types";

/**
 * The doctor-facing Results workspace for the most recently analysed note.
 *
 * Every block is tagged with where its content comes from, using the
 * provenance kinds below, so a reader never has to guess whether something was
 * counted from historical records, generated as a suggestion, produced by the
 * model, or inferred by keyword rules.
 */

type Provenance = "historical" | "ai" | "model" | "interpretation";

const PROVENANCE: Record<Provenance, { label: string; className: string; help: string }> = {
  historical: {
    label: "Historical records",
    className: "border-teal-200 bg-teal-50 text-teal-800",
    help: "Counted or quoted from previous cases in the case library.",
  },
  ai: {
    label: "AI suggestion",
    className: "border-orange-300 bg-orange-50 text-orange-800",
    help: "Generated automatically by MedXplain from published external guidelines and gaps in this note. Not historical records and not treatment advice.",
  },
  model: {
    label: "Model prediction",
    className: "border-violet-200 bg-violet-50 text-violet-800",
    help: "Output of the trained risk model and its explanation method.",
  },
  interpretation: {
    label: "Interpretation",
    className: "border-amber-200 bg-amber-50 text-amber-800",
    help: "Inferred by keyword rules; not stated in the note. Verify before relying on it.",
  },
};

const NOT_DOCUMENTED = "Not documented in the provided note";

interface Props {
  result: PredictResponse | null;
  clinical: ClinicalAnalysis | null;
  /** The exact text that was analysed; evidence offsets index into this. */
  analysedNote: string;
  /** What is currently in the editor, to warn when it no longer matches. */
  currentNote: string;
  loading: boolean;
  predictError: string | null;
  clinicalError: string | null;
  onGoToAnalyse: () => void;
  onOpenPeerConsult: () => void;
}

export function ClinicalWorkspace({
  result,
  clinical,
  analysedNote,
  currentNote,
  loading,
  predictError,
  clinicalError,
  onGoToAnalyse,
  onOpenPeerConsult,
}: Props) {
  const hasCase = Boolean(analysedNote) && (loading || result || clinical || predictError || clinicalError);
  const edited = Boolean(analysedNote) && currentNote.trim() !== analysedNote.trim();

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <WorkspaceHeader clinical={clinical} analysedNote={analysedNote} hasCase={Boolean(hasCase)} />

      {!hasCase && <NoCaseState onGoToAnalyse={onGoToAnalyse} />}

      {hasCase && (
        <>
          {edited && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
              The note in the editor has changed since this analysis. Everything below refers to the
              version that was analysed. Press <strong>Analyse note</strong> again to update it.
            </div>
          )}

          {loading && !clinical && <SectionSkeleton title="Reading the note…" />}
          {clinicalError && (
            <ErrorCard
              title="The previous-case search could not be completed"
              body={clinicalError}
            />
          )}

          {clinical && (
            <>
              <WorkspaceSection
                id="previous"
                title="Clinical Insights from Previous Cases"
                provenance={["historical", "ai", "interpretation"]}
              >
                <PreviousCases data={clinical.similar_cases} />
                <AiSuggestions items={clinical.ai_suggestions ?? []} />
              </WorkspaceSection>
            </>
          )}

          <WorkspaceSection id="risk" title="Patient-Specific Risk Assessment" provenance={["model"]}>
            <RiskAssessment
              result={result}
              loading={loading}
              error={predictError}
              onGoToAnalyse={onGoToAnalyse}
            />
          </WorkspaceSection>

          <WorkspaceSection id="peer" title="Need Another Clinical Perspective?">
            <PeerConsultPanel onOpen={onOpenPeerConsult} />
          </WorkspaceSection>
        </>
      )}

      <TechnicalDetails />

      <p className="px-1 text-xs leading-relaxed text-slate-400">
        {clinical?.disclaimer ??
          "Research prototype. Not a medical device and not clinically validated. All content must be verified by a clinician."}
      </p>
    </div>
  );
}

// -----------------------------------------------------------------------------------
// Frame
// -----------------------------------------------------------------------------------

function WorkspaceHeader({
  clinical,
  analysedNote,
  hasCase,
}: {
  clinical: ClinicalAnalysis | null;
  analysedNote: string;
  hasCase: boolean;
}) {
  return (
    <section className="card p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Clinical workspace</h2>
          <p className="text-sm text-slate-500">
            Comparable previous cases, suggestions and the model&rsquo;s risk assessment for the note you analysed.
          </p>
        </div>
        {hasCase && clinical && (
          <div className="text-right text-xs text-slate-500">
            <p>
              Case <span className="font-mono text-slate-700">{clinical.case_id}</span>
            </p>
            <p>Analysed {new Date(clinical.analysed_at * 1000).toLocaleTimeString()}</p>
          </div>
        )}
      </div>
      {hasCase && analysedNote && (
        <p className="mt-3 line-clamp-2 rounded-lg bg-slate-50 px-3 py-2 font-mono text-xs text-slate-600">
          {analysedNote.trim().slice(0, 240)}
          {analysedNote.trim().length > 240 ? "…" : ""}
        </p>
      )}
      <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
        <span>Labels used on this page:</span>
        {(Object.keys(PROVENANCE) as Provenance[]).map((p) => (
          <ProvenanceBadge key={p} kind={p} withHelp />
        ))}
      </div>
    </section>
  );
}

function WorkspaceSection({
  id,
  title,
  provenance = [],
  children,
}: {
  id: string;
  title: string;
  provenance?: Provenance[];
  children: React.ReactNode;
}) {
  return (
    <section className="card" aria-labelledby={`ws-${id}`}>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-5 py-3">
        <h2 id={`ws-${id}`} className="text-base font-semibold text-slate-900">
          {title}
        </h2>
        <div className="flex flex-wrap gap-1.5">
          {provenance.map((p) => (
            <ProvenanceBadge key={p} kind={p} />
          ))}
        </div>
      </div>
      <div className="p-5">{children}</div>
    </section>
  );
}

function ProvenanceBadge({ kind, withHelp = false }: { kind: Provenance; withHelp?: boolean }) {
  const p = PROVENANCE[kind];
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${p.className}`}
      title={p.help}
    >
      {p.label}
      {withHelp && <span className="sr-only">: {p.help}</span>}
    </span>
  );
}

function NoCaseState({ onGoToAnalyse }: { onGoToAnalyse: () => void }) {
  return (
    <section className="card grid min-h-[260px] place-items-center p-8 text-center">
      <div className="max-w-md">
        <h2 className="text-base font-semibold text-slate-800">No note analysed yet</h2>
        <p className="mt-1.5 text-sm leading-relaxed text-slate-500">
          The clinical workspace summarises the note you analyse. Paste a discharge summary on the
          Analyse tab and press <strong>Analyse note</strong>, then come back here.
        </p>
        <button
          type="button"
          onClick={onGoToAnalyse}
          className="mt-4 rounded-lg bg-clinical-600 px-4 py-2 text-sm font-medium text-white hover:bg-clinical-700"
        >
          Go to Analyse
        </button>
      </div>
    </section>
  );
}

function SectionSkeleton({ title }: { title: string }) {
  return (
    <section className="card p-5">
      <p className="mb-3 text-sm text-slate-500">{title}</p>
      <div className="space-y-2">
        <div className="skeleton h-4 w-2/3" />
        <div className="skeleton h-4 w-full" />
        <div className="skeleton h-4 w-5/6" />
      </div>
    </section>
  );
}

function ErrorCard({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-4">
      <p className="text-sm font-medium text-red-800">{title}</p>
      <p className="mt-1 text-xs text-red-700">{body}</p>
    </div>
  );
}

function Missing({ children = NOT_DOCUMENTED }: { children?: React.ReactNode }) {
  return <p className="text-sm italic text-slate-400">{children}</p>;
}

// -----------------------------------------------------------------------------------
// Previous cases
// -----------------------------------------------------------------------------------

function PreviousCases({ data }: { data: SimilarCases }) {
  if (!data.available) {
    return (
      <div className="rounded-lg border border-dashed border-slate-300 p-5 text-sm text-slate-600">
        <p className="font-medium text-slate-800">No historical case library is available</p>
        {data.reason && <p className="mt-1 text-xs text-slate-500">Reason: {data.reason}</p>}
        {data.data_needed && (
          <p className="mt-2">
            <strong>What is needed:</strong> {data.data_needed}
          </p>
        )}
        <p className="mt-2 text-xs text-slate-500">
          No comparable cases, outcomes or recommendations are shown, because none could be computed.
        </p>
      </div>
    );
  }

  const n = data.n_matched ?? 0;
  return (
    <div className="space-y-4">
      {data.source.synthetic && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
          <strong>Synthetic demonstration data.</strong> The case library contains computer-generated
          notes, not real patients. The counts below show how the tool works; they are not clinical
          evidence.
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-3">
        <Tile label="Matching previous cases" value={n.toLocaleString()} sub={`out of ${data.library_size?.toLocaleString()} in the library`} />
        {data.outcomes?.map((o) => (
          <Tile
            key={o.key}
            label={o.label}
            value={`${o.count} of ${o.n}`}
            sub={`${(o.rate * 100).toFixed(0)}% in these cases · ${(o.library_rate * 100).toFixed(0)}% across the whole library`}
          />
        ))}
      </div>

      <div className="rounded-lg border border-slate-200 p-3">
        <p className="mb-1.5 text-sm font-medium text-slate-700">How cases were matched</p>
        <ul className="space-y-1 text-sm text-slate-600">
          {data.criteria?.map((c, i) => (
            <li key={i}>
              <span className="font-medium text-slate-800">{c.label}</span>
              {c.detail && <span className="text-slate-500"> · {c.detail}</span>}
            </li>
          ))}
        </ul>
        {(data.excluded_duplicates ?? 0) > 0 && (
          <p className="mt-2 text-xs text-slate-500">
            {data.excluded_duplicates} near-identical record(s) were excluded, because the note you
            submitted appears to be from the library itself.
          </p>
        )}
      </div>

      {n === 0 ? (
        <Missing>{data.reason ?? "No previous case met the matching criteria."}</Missing>
      ) : (
        <>
          <RecommendationTable data={data} />
          <ExampleCases data={data} />
        </>
      )}

      <p className="rounded-lg bg-slate-50 p-3 text-xs leading-relaxed text-slate-500">
        Outcomes are what was recorded for the matched cases. They describe those patients, not this
        one, and differences between groups may reflect who received which care rather than its
        effect.
      </p>
    </div>
  );
}

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-slate-200 p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-0.5 text-xl font-semibold text-slate-900">{value}</p>
      {sub && <p className="text-[11px] text-slate-500">{sub}</p>}
    </div>
  );
}

function RecommendationTable({ data }: { data: SimilarCases }) {
  const rows = data.recommendations ?? [];
  return (
    <div className="rounded-lg border border-slate-200">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-3 py-2">
        <p className="text-sm font-medium text-slate-700">Recommendations documented in those cases</p>
        <ProvenanceBadge kind="historical" />
      </div>
      <p className="px-3 pt-2 text-xs text-slate-500">
        Quoted from clinicians&rsquo; notes in the matched cases and grouped by keyword. These are{" "}
        <strong>historical documentation, not AI suggestions</strong>. How often something was
        documented does not show that it was effective.
      </p>
      {rows.length === 0 ? (
        <div className="p-3">
          <Missing>No recommendation statements were found in the matched cases.</Missing>
        </div>
      ) : (
        <ul className="divide-y divide-slate-100">
          {rows.map((r) => (
            <RecommendationRow key={r.category} row={r} />
          ))}
        </ul>
      )}
    </div>
  );
}

function RecommendationRow({ row }: { row: NonNullable<SimilarCases["recommendations"]>[number] }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="px-3 py-2">
      <div className="flex flex-wrap items-center gap-3">
        <span className="min-w-0 flex-1 text-sm text-slate-800">{row.category}</span>
        <div className="flex w-48 items-center gap-2">
          <div className="h-2 flex-1 rounded-full bg-slate-100">
            <div className="h-2 rounded-full bg-teal-500" style={{ width: `${row.frequency * 100}%` }} />
          </div>
          <span className="w-16 text-right text-xs text-slate-600">
            {row.n_cases} of {row.n}
          </span>
        </div>
        <button type="button" onClick={() => setOpen((o) => !o)} className="text-xs text-clinical-600 hover:underline">
          {open ? "Hide quotes" : "Show quotes"}
        </button>
      </div>
      {open && (
        <ul className="mt-2 space-y-1">
          {row.examples.map((q, i) => (
            <li key={i} className="rounded-md border-l-2 border-teal-300 bg-teal-50/50 px-3 py-1.5 text-sm text-slate-700">
              &ldquo;{q.text}&rdquo; <span className="font-mono text-[11px] text-slate-400">{q.note_id}</span>
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

function ExampleCases({ data }: { data: SimilarCases }) {
  const [openId, setOpenId] = useState<string | null>(null);
  const rows = data.examples ?? [];
  if (rows.length === 0) return null;
  return (
    <div className="rounded-lg border border-slate-200">
      <p className="border-b border-slate-100 px-3 py-2 text-sm font-medium text-slate-700">
        Closest matching cases
      </p>
      <ul className="divide-y divide-slate-100">
        {rows.map((c) => (
          <li key={c.note_id} className="px-3 py-2 text-sm">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <span className="font-mono text-xs text-slate-500">{c.note_id}</span>
              <span className="text-slate-700">
                {c.age ?? "?"} y, {c.sex ?? "?"}
              </span>
              <span
                className={`chip ${c.died_in_hospital ? "bg-red-100 text-red-800" : "bg-emerald-100 text-emerald-800"}`}
              >
                {c.died_in_hospital ? "Died in hospital" : "Discharged alive"}
              </span>
              {c.readmitted_30d && !c.died_in_hospital && (
                <span className="chip bg-amber-100 text-amber-800">Readmitted ≤30 days</span>
              )}
              {c.disposition && (
                <span className="text-xs text-slate-500" title="Category matched from the disposition text of that note; the recorded outcome comes from the structured outcome field.">
                  Disposition written in note: {c.disposition}
                </span>
              )}
              <span className="text-xs text-slate-400">text similarity {c.similarity.toFixed(2)}</span>
              <button
                type="button"
                onClick={() => setOpenId(openId === c.note_id ? null : c.note_id)}
                className="ml-auto text-xs text-clinical-600 hover:underline"
              >
                {openId === c.note_id ? "Hide note" : "View source note"}
              </button>
            </div>
            {c.shared_conditions.length > 0 && (
              <p className="mt-1 text-xs text-slate-500">Shared conditions: {c.shared_conditions.join(", ")}</p>
            )}
            {openId === c.note_id && (
              <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap rounded bg-slate-50 p-3 font-mono text-[12px] text-slate-600">
                {c.text}
              </pre>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

// -----------------------------------------------------------------------------------
// Risk assessment
// -----------------------------------------------------------------------------------

function RiskAssessment({
  result,
  loading,
  error,
  onGoToAnalyse,
}: {
  result: PredictResponse | null;
  loading: boolean;
  error: string | null;
  onGoToAnalyse: () => void;
}) {
  const [active, setActive] = useState(0);
  if (error) return <ErrorCard title="The risk model could not score this note" body={error} />;
  if (!result) {
    return loading ? (
      <div className="space-y-2">
        <div className="skeleton h-24 w-full" />
        <p className="text-xs text-slate-500">Running the risk model and its explanation…</p>
      </div>
    ) : (
      <Missing>No model prediction is available for this note.</Missing>
    );
  }
  const calibrated = isCalibrated(result);
  const flagged = result.predicted_label === 1;
  const explanation = result.explanations[Math.min(active, result.explanations.length - 1)];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-slate-200 p-3">
        <span className={`chip ${flagged ? "bg-red-100 text-red-800" : "bg-emerald-100 text-emerald-800"}`}>
          {flagged ? "Flagged by the model" : "Not flagged by the model"}
        </span>
        <span className="text-sm text-slate-700">
          {result.model_display_name}: {flagged ? "above" : "below"} its alert level for {result.outcome}
          {calibrated && ` (${(result.probability * 100).toFixed(1)}%)`}.
        </span>
        <button type="button" onClick={onGoToAnalyse} className="ml-auto text-xs text-clinical-600 hover:underline">
          Full prediction and highlighted note on the Analyse tab →
        </button>
      </div>

      <div
        className={`rounded-lg border p-3 text-sm ${
          calibrated ? "border-emerald-200 bg-emerald-50 text-emerald-900" : "border-amber-200 bg-amber-50 text-amber-900"
        }`}
      >
        <p className="font-medium">{calibrated ? "Calibration checked" : "Shown as a ranking, not a percentage"}</p>
        <p className="mt-0.5 text-xs leading-relaxed">
          {result.calibration?.summary ?? "The API did not report calibration evidence for this model."}
        </p>
      </div>

      {explanation && (
        <>
          <p className="text-xs leading-relaxed text-slate-500">
            The words below show how the model reacted to the text. They are not evidence that a finding
            causes or prevents death. Attention-based methods show only where the model looked, not
            whether a word raised or lowered its score.
          </p>
          {result.explanations.length > 1 && (
            <div className="flex flex-wrap gap-1.5">
              {result.explanations.map((e, i) => (
                <button
                  key={e.method}
                  type="button"
                  onClick={() => setActive(i)}
                  className={`chip border ${
                    i === active
                      ? "border-clinical-500 bg-clinical-50 text-clinical-700"
                      : "border-slate-300 bg-white text-slate-600"
                  }`}
                >
                  {e.method}
                </button>
              ))}
            </div>
          )}
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="rounded-lg border border-slate-200">
              <p className="border-b border-slate-100 px-3 py-2 text-sm font-medium text-slate-700">
                Top contributing words
              </p>
              <div className="p-2">
                <ImportanceChart explanation={explanation} />
              </div>
            </div>
            <div className="rounded-lg border border-slate-200">
              <p className="border-b border-slate-100 px-3 py-2 text-sm font-medium text-slate-700">
                How to read this
              </p>
              <HowToRead explanation={explanation} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// -----------------------------------------------------------------------------------
// AI suggestions (kept visually separate from historical findings)
// -----------------------------------------------------------------------------------

function AiSuggestions({ items }: { items: ClinicalAnalysis["ai_suggestions"] }) {
  return (
    <div className="mt-4 rounded-lg border-2 border-orange-300 bg-orange-50 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-sm font-semibold text-orange-900">AI suggestions</p>
        <ProvenanceBadge kind="ai" />
      </div>
      <p className="mt-0.5 text-xs text-orange-800">
        Generated automatically by MedXplain from published external guidelines that match conditions
        mentioned in this note, plus gaps in its documentation. They are not drawn from the historical
        cases above and are not treatment advice. Check the guidance yourself and use your clinical
        judgement.
      </p>
      {items.length === 0 ? (
        <p className="mt-2 text-sm italic text-orange-800">No suggestions for this note.</p>
      ) : (
        <ul className="mt-2 space-y-1.5">
          {items.map((s, i) => (
            <li key={i} className="text-sm text-orange-950">
              <span aria-hidden className="mr-1.5">•</span>
              {s.text}
              {s.sources && s.sources.length > 0 && (
                <span className="block pl-4">
                  {s.sources.map((src) => (
                    <a
                      key={src.url}
                      href={src.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mr-3 inline-block text-[12px] font-medium text-orange-900 underline decoration-orange-400 underline-offset-2 hover:text-orange-700"
                    >
                      {src.publisher}: {src.title} ↗
                    </a>
                  ))}
                </span>
              )}
              <span className="block pl-4 text-[11px] text-orange-700">Why: {s.basis}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// -----------------------------------------------------------------------------------
// Peer consult
// -----------------------------------------------------------------------------------

/**
 * Entry point to the Peer Consult tab, which opens with the analysed note
 * prefilled. Peer Consult is frontend-only, so the note never leaves the browser.
 */
export function PeerConsultPanel({ onOpen }: { onOpen: () => void }) {
  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-700">
        Explore additional clinical perspectives on this case in Peer Consult.
      </p>
      <button
        type="button"
        onClick={onOpen}
        className="rounded-lg bg-clinical-600 px-4 py-2 text-sm font-medium text-white hover:bg-clinical-700"
      >
        Ask a Peer Physician
      </button>
      <p className="text-xs text-slate-500">
        Opens Peer Consult with this note filled in. The responses there are simulated for
        demonstration, and nothing leaves your browser.
      </p>
    </div>
  );
}

// -----------------------------------------------------------------------------------
// Technical details
// -----------------------------------------------------------------------------------

function TechnicalDetails() {
  const [open, setOpen] = useState(false);
  return (
    <details
      className="card"
      onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
    >
      <summary className="cursor-pointer select-none px-5 py-3 text-base font-semibold text-slate-900 hover:text-clinical-700">
        Technical Details: Model Evaluation and Validation
        <span className="ml-2 text-xs font-normal text-slate-500">
          Model scores, datasets and explanation quality (for technical readers)
        </span>
      </summary>
      {/* mounted only when open so charts measure a visible container */}
      {open && (
        <div className="border-t border-slate-100 p-4">
          <ResultsView />
        </div>
      )}
    </details>
  );
}
