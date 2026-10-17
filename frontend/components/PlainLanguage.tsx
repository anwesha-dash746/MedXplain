"use client";

import type { ExplanationPayload, PredictResponse } from "@/lib/types";

/**
 * Plain-language companions to the prediction and explanation cards.
 *
 * The numeric cards are accurate but assume the reader knows what a decision
 * threshold or an attribution method is. These components say the same thing
 * in everyday words, and keep the technical figures one click away rather than
 * removing them.
 */

const BAND_STYLE: Record<string, { bg: string; border: string; text: string; icon: string }> = {
  low: { bg: "bg-emerald-50", border: "border-emerald-200", text: "text-emerald-900", icon: "✓" },
  borderline: { bg: "bg-amber-50", border: "border-amber-200", text: "text-amber-900", icon: "!" },
  elevated: { bg: "bg-orange-50", border: "border-orange-200", text: "text-orange-900", icon: "!" },
  high: { bg: "bg-red-50", border: "border-red-200", text: "text-red-900", icon: "!!" },
};

const BAND_HEADLINE: Record<string, string> = {
  low: "Lower risk: the model did not flag this patient",
  borderline: "Close to the line: worth a second look",
  elevated: "Flagged: higher than usual risk",
  high: "Flagged: high risk",
};

const BAND_MEANING: Record<string, string> = {
  low:
    "The note reads like those of patients who usually survived their hospital stay. " +
    "Lower risk does not mean no risk, so keep using your clinical judgement.",
  borderline:
    "The estimate is below the model's alert level but not far from it. Small details in " +
    "the note could tip it either way, so it is worth checking the highlighted words below.",
  elevated:
    "The estimate is above the level at which the model raises an alert. The note shares " +
    "features with patients who did not survive their stay. Consider whether closer " +
    "monitoring or a review of the care plan is appropriate.",
  high:
    "The estimate is well above the model's alert level. The note strongly resembles those " +
    "of patients who did not survive their stay. Consider prompt clinical review.",
};

/**
 * A score is shown as a percentage chance only when the evaluation files show it
 * is calibrated. Missing evidence counts as "not calibrated", never as a pass.
 */
export function isCalibrated(result: PredictResponse): boolean {
  return result.calibration?.well_calibrated === true;
}

function pct(x: number, digits = 1) {
  return `${(x * 100).toFixed(digits)}%`;
}

/** "about 3 in 10" style natural frequency, which most people read more easily than a percentage. */
function naturalFrequency(p: number): string {
  if (p < 0.01) return "fewer than 1 in 100";
  if (p >= 0.995) return "almost all";
  if (p >= 0.1) return `about ${Math.round(p * 10)} in 10`;
  return `about ${Math.round(p * 100)} in 100`;
}

export function PlainRiskSummary({ result }: { result: PredictResponse }) {
  const style = BAND_STYLE[result.risk_band] ?? BAND_STYLE.borderline;
  const outcome = result.outcome.toLowerCase();
  const flagged = result.predicted_label === 1;
  const ratio = result.probability / Math.max(result.decision_threshold, 1e-9);
  const calibrated = isCalibrated(result);

  return (
    <div className="space-y-3">
      <div className={`rounded-lg border p-4 ${style.bg} ${style.border} ${style.text}`}>
        <p className="flex items-center gap-2 text-sm font-semibold">
          <span
            aria-hidden
            className="inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-white/70 px-1 text-[11px] font-bold"
          >
            {style.icon}
          </span>
          {BAND_HEADLINE[result.risk_band] ?? "Risk estimate"}
        </p>
        <p className="mt-2 text-sm leading-relaxed">{BAND_MEANING[result.risk_band]}</p>
      </div>

      <div className="rounded-lg border border-slate-200 p-4 text-sm leading-relaxed text-slate-700">
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">In plain words</p>
        <ul className="list-disc space-y-1.5 pl-5">
          {calibrated ? (
            <li>
              Out of 100 patients whose notes look like this one, the model expects{" "}
              <strong>{naturalFrequency(result.probability)}</strong> to die before leaving hospital (
              {pct(result.probability)} chance of {outcome}).
            </li>
          ) : (
            <li>
              The model&rsquo;s score is shown <strong>without a percentage</strong>.{" "}
              {result.calibration?.evaluated
                ? "When it was tested on held-out notes, its scores did not match the death rates actually observed, so the number cannot be read as a chance of dying."
                : "Its scores have not been checked against observed death rates, so the number cannot be read as a chance of dying."}{" "}
              Use it only to see whether the note is above or below the alert level.
            </li>
          )}
          <li>
            The model raises an alert once its score reaches{" "}
            <strong>{calibrated ? pct(result.decision_threshold) : "the alert level chosen during tuning"}</strong>.
            This patient is{" "}
            {flagged ? (
              <strong>above that level{calibrated && ratio >= 1.5 ? `, about ${ratio.toFixed(1)}× the alert level` : ""}</strong>
            ) : (
              <strong>below that level</strong>
            )}
            .
          </li>
          {!calibrated ? null : result.decision_threshold < 0.3 ? (
            <li>
              The alert level is set low because in-hospital death is uncommon. Even a risk that
              sounds small can be well above average, so go by the alert, not just the size of the
              percentage.
            </li>
          ) : (
            <li>
              The alert level is not 50%. It was chosen to balance catching at-risk patients
              against raising too many false alarms, so go by the alert rather than whether the
              number is above or below half.
            </li>
          )}
        </ul>
      </div>

      <RiskScale result={result} calibrated={calibrated} />
    </div>
  );
}

/** A simple bar from 0 to 100% with the patient's estimate and the alert level marked. */
function RiskScale({ result, calibrated }: { result: PredictResponse; calibrated: boolean }) {
  const p = Math.min(Math.max(result.probability, 0), 1);
  const t = Math.min(Math.max(result.decision_threshold, 0), 1);
  return (
    <div className="px-1">
      <div className="relative h-3 rounded-full bg-gradient-to-r from-emerald-200 via-amber-200 to-red-300">
        <div
          className="absolute -top-1 h-5 w-0.5 bg-slate-900"
          style={{ left: `${t * 100}%` }}
          title={calibrated ? `Alert level ${pct(t)}` : "Alert level"}
        />
        <div
          className="absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-slate-900 shadow"
          style={{ left: `${p * 100}%` }}
          title={calibrated ? `This patient ${pct(p)}` : "This patient's model score"}
        />
      </div>
      <div className="mt-1.5 flex justify-between text-[11px] text-slate-500">
        {calibrated ? (
          <>
            <span>0% · very unlikely</span>
            <span>100% · almost certain</span>
          </>
        ) : (
          <>
            <span>lowest model score</span>
            <span>highest model score</span>
          </>
        )}
      </div>
      <p className="mt-1 text-[11px] text-slate-500">
        <span className="mr-1 inline-block h-2.5 w-2.5 rounded-full bg-slate-900 align-middle" /> this patient
        <span className="mx-1.5 ml-3 inline-block h-3 w-0.5 bg-slate-900 align-middle" /> alert level
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------

export const METHOD_FRIENDLY: Record<string, { name: string; what: string }> = {
  attention_rollout: {
    name: "Attention (where the model looked)",
    what:
      "This shows which words the model paid the most attention to while reading the note. " +
      "It tells you what the model focused on, but not whether each word made the risk go up or down.",
  },
  attention_raw: {
    name: "Attention, last layer only",
    what:
      "A quick view of which words the model focused on in its final reading step. It is fast " +
      "but rough, and it does not say whether a word raised or lowered the risk.",
  },
  attention_grad: {
    name: "Attention weighted toward this prediction",
    what:
      "This shows the words the model focused on, weighted by how much they mattered to this " +
      "risk prediction in particular. It still does not separate words that raised the risk from words that lowered it.",
  },
  shap: {
    name: "SHAP (each word's share of the risk)",
    what:
      "Each word gets a share of the credit for the model's final score. Red words pushed the model's score up and " +
      "blue words pulled it down. This is the most thorough method here and also the slowest.",
  },
  occlusion: {
    name: "Remove-a-word test",
    what:
      "The model re-reads the note with each word hidden in turn. If hiding a word makes the score " +
      "drop, that word was raising it (red). If hiding it makes the score rise, it was lowering it (blue).",
  },
  integrated_gradients: {
    name: "Integrated Gradients",
    what:
      "This traces how the risk builds up from a blank note to the full note and credits each word " +
      "with its part in that change. Red words raised the model's score and blue words lowered it.",
  },
};

export function HowToRead({ explanation }: { explanation: ExplanationPayload }) {
  const friendly = METHOD_FRIENDLY[explanation.method] ?? {
    name: explanation.method,
    what: explanation.description,
  };
  const ups = explanation.top_positive.slice(0, 3).map((w) => w.text);
  const downs = explanation.top_negative.slice(0, 3).map((w) => w.text);

  return (
    <div className="space-y-4 p-4 text-sm leading-relaxed text-slate-700">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Method used</p>
        <p className="font-medium text-slate-800">{friendly.name}</p>
        <p className="mt-1">{friendly.what}</p>
      </div>

      <ol className="space-y-2">
        <Step n={1} title="Look at the coloured words in the note">
          {explanation.signed ? (
            <>
              <span className="rounded bg-red-200 px-1">Red</span> words pushed the <em>model&rsquo;s score</em>{" "}
              <strong>up</strong>.{" "}
              <span className="rounded bg-blue-200 px-1">Blue</span> words pulled it{" "}
              <strong>down</strong>. Uncoloured words made little difference to the score. This
              describes the model&rsquo;s behaviour, not what causes risk in the patient.
            </>
          ) : (
            <>
              <span className="rounded bg-violet-200 px-1">Purple</span> words are the ones the
              model focused on. Purple only means &ldquo;this mattered&rdquo;. It does{" "}
              <strong>not</strong> say whether the word raised or lowered the risk.
            </>
          )}
        </Step>
        <Step n={2} title="Darker means stronger">
          The darkest word is the one that mattered most <em>in this note</em>. Shading is relative,
          so compare words within one note rather than across different notes.
        </Step>
        <Step n={3} title="Check the bar chart">
          It lists the most influential words from strongest to weakest, so you can see them at a
          glance. Hover over a word in the note to see its exact score.
        </Step>
        <Step n={4} title="Does it make clinical sense?">
          If the model relies on meaningful findings (for example shock, intubation or rising
          lactate), that is reassuring. If it relies on words with no clinical meaning, treat the
          prediction with more caution.
        </Step>
      </ol>

      {(ups.length > 0 || downs.length > 0) && explanation.signed && (
        <div className="rounded-lg bg-slate-50 p-3 text-xs">
          {ups.length > 0 && (
            <p>
              <strong className="text-red-700">Raised the model&rsquo;s score most:</strong> {ups.join(", ")}
            </p>
          )}
          {downs.length > 0 && (
            <p className="mt-1">
              <strong className="text-blue-700">Lowered the model&rsquo;s score most:</strong> {downs.join(", ")}
            </p>
          )}
        </div>
      )}
      {!explanation.signed && ups.length > 0 && (
        <div className="rounded-lg bg-slate-50 p-3 text-xs">
          <strong className="text-violet-700">Words the model focused on most:</strong> {ups.join(", ")}
        </div>
      )}

      <p className="rounded-lg border border-slate-200 p-3 text-xs text-slate-500">
        These highlights show how the <em>model</em> reasoned. They do not prove anything about the
        patient. A word can be highlighted simply because it often appears in notes of sicker
        patients.
      </p>
    </div>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-clinical-50 text-[11px] font-semibold text-clinical-700">
        {n}
      </span>
      <div>
        <p className="font-medium text-slate-800">{title}</p>
        <p className="text-[13px] text-slate-600">{children}</p>
      </div>
    </li>
  );
}
