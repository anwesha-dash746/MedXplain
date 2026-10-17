"use client";

import { useState } from "react";
import { HighlightedNote, HighlightLegend } from "./HighlightedNote";
import { ImportanceChart } from "./ImportanceChart";
import { RiskGauge } from "./RiskGauge";
import { HowToRead, PlainRiskSummary, isCalibrated } from "./PlainLanguage";
import type { PredictResponse } from "@/lib/types";

interface Props {
  result: PredictResponse;
  noteText: string;
}

export function ResultPanel({ result, noteText }: Props) {
  const [activeIndex, setActiveIndex] = useState(0);
  const explanation = result.explanations[Math.min(activeIndex, result.explanations.length - 1)];
  const comparing = result.explanations.length > 1;

  return (
    <div className="space-y-4 animate-fade-in">
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Risk of dying during this hospital stay</h2>
          <span className="text-xs text-slate-400">
            {result.model_display_name} · {result.inference_ms.toFixed(0)} ms
          </span>
        </div>
        <div className="grid gap-4 p-5 md:grid-cols-[240px_1fr]">
          <RiskGauge result={result} />
          <div className="space-y-3">
            <PlainRiskSummary result={result} />
            <p className="rounded-lg bg-slate-50 p-3 text-xs leading-relaxed text-slate-600">
              <strong>Important:</strong> this is an estimate from a computer model trained on past
              hospital notes. It supports clinical decisions but does not make them, and it should
              always be read alongside the note itself and the patient in front of you.
            </p>
            {result.warnings.map((warning, i) => (
              <div
                key={i}
                className="flex gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-relaxed text-amber-800"
              >
                <span aria-hidden>⚠</span>
                <span>{warning}</span>
              </div>
            ))}
            <details className="group rounded-lg border border-slate-200 text-xs">
              <summary className="cursor-pointer select-none px-3 py-2 font-medium text-slate-500 hover:text-slate-700">
                Technical details
              </summary>
              <div className="space-y-2 border-t border-slate-100 p-3">
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  <Stat
                    label={isCalibrated(result) ? "Probability" : "Model score (uncalibrated)"}
                    value={`${(result.probability * 100).toFixed(1)}%`}
                  />
                  <Stat label="Threshold" value={`${(result.decision_threshold * 100).toFixed(1)}%`} />
                  <Stat label="Note tokens" value={result.n_tokens.toLocaleString()} />
                  <Stat label="Characters" value={result.n_chars.toLocaleString()} />
                </div>
                <p className="leading-relaxed text-slate-500">{result.calibration_note}</p>
                <p className="leading-relaxed text-slate-500">
                  <strong>Calibration:</strong>{" "}
                  {result.calibration?.summary ?? "Calibration status was not reported by the API."}
                  {result.calibration?.evaluated && (
                    <> (criteria: {result.calibration.criteria}; checkpoint seed {result.calibration.seed ?? "unknown"})</>
                  )}
                </p>
                <p className="leading-relaxed text-slate-500">
                  The threshold (alert level) was chosen on validation data, not on the test set.
                  &ldquo;Tokens&rdquo; are the word pieces the model reads.
                </p>
              </div>
            </details>
          </div>
        </div>
      </div>

      {explanation && (
        <>
          {comparing && (
            <div className="flex flex-wrap gap-1.5">
              {result.explanations.map((e, index) => (
                <button
                  key={e.method}
                  type="button"
                  onClick={() => setActiveIndex(index)}
                  className={`chip border transition-colors ${
                    index === activeIndex
                      ? "border-clinical-500 bg-clinical-50 text-clinical-700"
                      : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  {e.method}
                  <span className="ml-1 text-[10px] text-slate-400">
                    {e.compute_ms >= 1000 ? `${(e.compute_ms / 1000).toFixed(1)}s` : `${e.compute_ms.toFixed(0)}ms`}
                  </span>
                </button>
              ))}
            </div>
          )}

          <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
            <div className="card">
              <div className="card-header">
                <h2 className="card-title">Highlighted note</h2>
                <span className="font-mono text-[11px] text-slate-400">{explanation.method}</span>
              </div>
              <div className="border-b border-slate-100 px-5 py-2.5">
                <HighlightLegend signed={explanation.signed} />
              </div>
              <div className="max-h-[520px] overflow-y-auto p-2">
                <HighlightedNote text={noteText} explanation={explanation} />
              </div>
            </div>

            <div className="space-y-4">
              <div className="card">
                <div className="card-header">
                  <h2 className="card-title">Top contributing words</h2>
                </div>
                <div className="p-2">
                  <ImportanceChart explanation={explanation} />
                </div>
              </div>

              <div className="card">
                <div className="card-header">
                  <h2 className="card-title">How to read this</h2>
                </div>
                <HowToRead explanation={explanation} />
                <details className="border-t border-slate-100 text-xs text-slate-600">
                  <summary className="cursor-pointer select-none px-4 py-2 font-medium text-slate-500 hover:text-slate-700">
                    Technical details
                  </summary>
                  <div className="space-y-2 px-4 pb-4 leading-relaxed">
                  <p>{explanation.description}</p>
                  <dl className="grid grid-cols-2 gap-x-3 gap-y-1 border-t border-slate-100 pt-2 text-[11px]">
                    <dt className="text-slate-400">Model evaluations</dt>
                    <dd className="text-right font-mono text-slate-700">
                      {explanation.n_model_evaluations.toLocaleString()}
                    </dd>
                    <dt className="text-slate-400">Compute time</dt>
                    <dd className="text-right font-mono text-slate-700">
                      {explanation.compute_ms >= 1000
                        ? `${(explanation.compute_ms / 1000).toFixed(2)} s`
                        : `${explanation.compute_ms.toFixed(0)} ms`}
                    </dd>
                    <dt className="text-slate-400">Direction</dt>
                    <dd className="text-right font-mono text-slate-700">
                      {explanation.signed ? "signed" : "magnitude only"}
                    </dd>
                    {typeof explanation.meta.convergence_delta === "number" && (
                      <>
                        <dt className="text-slate-400">Completeness error</dt>
                        <dd className="text-right font-mono text-slate-700">
                          {(explanation.meta.relative_convergence_error as number)?.toFixed(3) ??
                            (explanation.meta.convergence_delta as number).toFixed(3)}
                        </dd>
                      </>
                    )}
                  </dl>
                  </div>
                </details>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 px-3 py-2">
      <p className="text-[10px] uppercase tracking-wide text-slate-400">{label}</p>
      <p className="font-mono text-sm font-semibold text-slate-800">{value}</p>
    </div>
  );
}
