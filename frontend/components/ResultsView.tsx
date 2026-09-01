"use client";

import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getResults } from "@/lib/api";
import type { ResultsPayload, SeedAggregate } from "@/lib/types";

const CURVE_COLORS = ["#265f8c", "#c53030", "#0f9d76", "#7c3aed", "#dd6b20", "#0891b2"];

/**
 * Shows the actual evaluation artefacts produced by the scripts in `scripts/`.
 * Every section renders only if its results file exists, so the page degrades to
 * "not run yet" instead of displaying placeholder numbers that could be mistaken
 * for findings.
 */
export function ResultsView() {
  const [data, setData] = useState<ResultsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getResults()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "could not load results"));
  }, []);

  if (error) return <Notice title="Results unavailable" body={error} />;
  if (!data) return <div className="card p-6"><div className="skeleton h-40 w-full" /></div>;

  const hasAny = (data.available ?? []).length > 0;
  if (!hasAny) {
    return (
      <Notice
        title="No results yet"
        body="Run the pipeline to populate this page: scripts/00_build_dataset.py, then 01, 02, 03 and 04."
      />
    );
  }

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      {data.dataset_report && <DatasetSection report={data.dataset_report} />}
      {(data.transformers || data.baselines) && (
        <PerformanceSection transformers={data.transformers} baselines={data.baselines} />
      )}
      {data.evaluation?.significance && <SignificanceSection evaluation={data.evaluation} />}
      {data.faithfulness && <FaithfulnessSection faithfulness={data.faithfulness} />}
      {data.stability && <StabilitySection stability={data.stability} />}
      {data.plausibility && <PlausibilitySection plausibility={data.plausibility} />}
    </div>
  );
}

function DatasetSection({ report }: { report: any }) {
  const splits = report.splits ?? {};
  return (
    <Section title="Dataset" subtitle={`${report.source} · outcome: ${report.primary_outcome}`}>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="py-2">Split</th>
              <th className="py-2 text-right">Notes</th>
              <th className="py-2 text-right">Positives</th>
              <th className="py-2 text-right">Positive rate</th>
            </tr>
          </thead>
          <tbody>
            {["train", "val", "test"].filter((s) => splits[s]).map((split) => {
              const row = splits[split];
              const label = report.primary_outcome;
              return (
                <tr key={split} className="border-b border-slate-100">
                  <td className="py-2 font-medium capitalize text-slate-700">{split}</td>
                  <td className="py-2 text-right font-mono">{row.n.toLocaleString()}</td>
                  <td className="py-2 text-right font-mono">{row[`${label}_positive`] ?? "—"}</td>
                  <td className="py-2 text-right font-mono">
                    {row[`${label}_rate`] != null ? `${(row[`${label}_rate`] * 100).toFixed(1)}%` : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {report.token_length && (
        <p className="mt-3 text-xs text-slate-500">
          Note length: median {Math.round(report.token_length.median)} tokens, 95th percentile{" "}
          {Math.round(report.token_length.p95)}, {(report.token_length.frac_over_512 * 100).toFixed(1)}%
          exceed 512 tokens (tokenizer: <span className="font-mono">{report.token_length.tokenizer}</span>).
        </p>
      )}
    </Section>
  );
}

function PerformanceSection({ transformers, baselines }: { transformers?: any; baselines?: any }) {
  const rows: { name: string; family: string; agg: Record<string, SeedAggregate> }[] = [];

  for (const [key, block] of Object.entries<any>(baselines?.models ?? {})) {
    if (block.aggregate) rows.push({ name: prettyBaseline(key), family: "TF-IDF baseline", agg: block.aggregate });
  }
  for (const [key, block] of Object.entries<any>(transformers?.models ?? {})) {
    if (block.aggregate) rows.push({ name: key, family: "transformer", agg: block.aggregate });
  }
  rows.sort((a, b) => (b.agg.auprc?.mean ?? 0) - (a.agg.auprc?.mean ?? 0));

  const curves = collectCurves(transformers, baselines);

  return (
    <Section
      title="Predictive performance"
      subtitle="Held-out test set · mean ± std across random seeds"
    >
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="py-2">Model</th>
              <th className="py-2 text-right">AUROC</th>
              <th className="py-2 text-right">AUPRC</th>
              <th className="py-2 text-right">F1</th>
              <th className="py-2 text-right">Brier</th>
              <th className="py-2 text-right">Seeds</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr
                key={row.name}
                className={`border-b border-slate-100 ${index === 0 ? "bg-emerald-50/50" : ""}`}
              >
                <td className="py-2">
                  <span className="font-medium text-slate-800">{row.name}</span>
                  <span className="ml-2 text-[10px] uppercase tracking-wide text-slate-400">
                    {row.family}
                  </span>
                </td>
                <MetricCell value={row.agg.auroc} />
                <MetricCell value={row.agg.auprc} bold />
                <MetricCell value={row.agg.f1} />
                <MetricCell value={row.agg.brier} />
                <td className="py-2 text-right font-mono text-xs text-slate-500">
                  {row.agg.auroc?.n_seeds ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {curves.length > 0 && (
        <div className="mt-5 grid gap-4 md:grid-cols-2">
          <CurvePlot title="ROC" curves={curves} xKey="fpr" yKey="tpr" xLabel="False positive rate" yLabel="True positive rate" diagonal />
          <CurvePlot title="Precision–Recall" curves={curves} xKey="recall" yKey="precision" xLabel="Recall" yLabel="Precision" />
        </div>
      )}
    </Section>
  );
}

function SignificanceSection({ evaluation }: { evaluation: any }) {
  const sig = evaluation.significance;
  const tests = sig?.tests ?? {};
  const holm = sig?.holm?.results ?? {};
  return (
    <Section
      title="Statistical comparison"
      subtitle={`Best model vs. each baseline · paired bootstrap on ${sig?.metric ?? "auprc"} · Holm-adjusted`}
    >
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="py-2">Comparison</th>
            <th className="py-2 text-right">Δ</th>
            <th className="py-2 text-right">95% CI</th>
            <th className="py-2 text-right">p (raw)</th>
            <th className="py-2 text-right">p (Holm)</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries<any>(tests).map(([name, test]) => {
            const boot = test.bootstrap ?? {};
            const adj = holm[name];
            return (
              <tr key={name} className="border-b border-slate-100">
                <td className="py-2 font-medium text-slate-700">vs. {prettyBaseline(name)}</td>
                <td className="py-2 text-right font-mono">{fmt(boot.difference, 4)}</td>
                <td className="py-2 text-right font-mono text-xs text-slate-500">
                  [{fmt(boot.ci95?.[0], 3)}, {fmt(boot.ci95?.[1], 3)}]
                </td>
                <td className="py-2 text-right font-mono">{fmtP(boot.p_value)}</td>
                <td className="py-2 text-right font-mono">
                  <span className={adj?.significant ? "font-semibold text-emerald-700" : "text-slate-500"}>
                    {fmtP(adj?.p_adjusted)}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Section>
  );
}

function FaithfulnessSection({ faithfulness }: { faithfulness: any }) {
  const methods = faithfulness.methods ?? {};
  const curves = faithfulness.curves ?? {};
  const entries = Object.entries<any>(methods);

  const curveData = buildFaithfulnessCurve(curves);

  return (
    <Section
      title="Explanation faithfulness"
      subtitle={`ERASER comprehensiveness / sufficiency · ${faithfulness.n_notes ?? "?"} test notes · ${faithfulness.model ?? ""}`}
    >
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="py-2">Method</th>
            <th className="py-2 text-right" title="Higher is better">Compreh. ↑</th>
            <th className="py-2 text-right" title="Lower is better">Suffic. ↓</th>
            <th className="py-2 text-right" title="Comprehensiveness above a random explanation">vs. random ↑</th>
            <th className="py-2 text-right" title="Lower is better">Deletion AUC ↓</th>
            <th className="py-2 text-right">Evals/note</th>
          </tr>
        </thead>
        <tbody>
          {entries
            .sort((a, b) => (b[1].comprehensiveness?.mean ?? 0) - (a[1].comprehensiveness?.mean ?? 0))
            .map(([method, block]) => (
              <tr key={method} className="border-b border-slate-100">
                <td className="py-2 font-mono text-xs text-slate-700">{method}</td>
                <MetricCell value={block.comprehensiveness} digits={3} bold />
                <MetricCell value={block.sufficiency} digits={3} />
                <MetricCell value={block.comprehensiveness_over_random} digits={3} />
                <td className="py-2 text-right font-mono">
                  {fmt(curves[method]?.deletion_auc?.mean, 3)}
                </td>
                <td className="py-2 text-right font-mono text-xs text-slate-500">
                  {block.n_model_evaluations ?? faithfulness.evaluations?.[method] ?? "—"}
                </td>
              </tr>
            ))}
        </tbody>
      </table>

      {curveData.length > 0 && (
        <div className="mt-5">
          <p className="mb-1 text-xs font-medium text-slate-600">
            Deletion curves — confidence as the highest-scoring words are progressively masked.
            A faithful explanation drops fastest.
          </p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curveData} margin={{ top: 8, right: 16, bottom: 24, left: 0 }}>
                <CartesianGrid stroke="#f1f5f9" />
                <XAxis
                  dataKey="fraction"
                  tick={{ fontSize: 11, fill: "#64748b" }}
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  label={{ value: "fraction of tokens masked", position: "insideBottom", offset: -14, fontSize: 11, fill: "#64748b" }}
                />
                <YAxis tick={{ fontSize: 11, fill: "#64748b" }} domain={[0, 1]} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(v: number) => v.toFixed(3)} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {Object.keys(curves).map((method, i) => (
                  <Line
                    key={method}
                    type="monotone"
                    dataKey={method}
                    stroke={CURVE_COLORS[i % CURVE_COLORS.length]}
                    dot={false}
                    strokeWidth={2}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </Section>
  );
}

function StabilitySection({ stability }: { stability: any }) {
  const methods = stability.methods ?? {};
  return (
    <Section
      title="Explanation stability"
      subtitle={`Agreement under meaning-preserving edits · ${stability.n_notes ?? "?"} notes`}
    >
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="py-2">Method</th>
            <th className="py-2 text-right">Paraphrase ρ</th>
            <th className="py-2 text-right">Top-10 Jaccard</th>
            <th className="py-2 text-right">Near-dup ρ</th>
            <th className="py-2 text-right" title="Importance mass landing on an inserted irrelevant sentence">Distractor mass ↓</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries<any>(methods).map(([method, block]) => (
            <tr key={method} className="border-b border-slate-100">
              <td className="py-2 font-mono text-xs text-slate-700">{method}</td>
              <MetricCell value={block.paraphrase?.spearman} digits={3} bold />
              <MetricCell value={block.paraphrase?.top_k_jaccard} digits={3} />
              <MetricCell value={block.near_duplicate?.spearman} digits={3} />
              <MetricCell value={block.distractor?.distractor_mass} digits={3} />
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  );
}

function PlausibilitySection({ plausibility }: { plausibility: any }) {
  const methods = plausibility.methods ?? {};
  return (
    <Section
      title="Explanation plausibility"
      subtitle="Agreement with the gold rationale spans recorded by the corpus generator"
    >
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="py-2">Method</th>
            <th className="py-2 text-right">Token F1</th>
            <th className="py-2 text-right">Rationale AUPRC</th>
            <th className="py-2 text-right">Weight ρ</th>
            <th className="py-2 text-right">Sign agreement</th>
            <th className="py-2 text-right" title="Share of importance outside any true risk factor">Off-rationale ↓</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries<any>(methods).map(([method, block]) => (
            <tr key={method} className="border-b border-slate-100">
              <td className="py-2 font-mono text-xs text-slate-700">{method}</td>
              <MetricCell value={block.token_f1} digits={3} bold />
              <MetricCell value={block.rationale_auprc} digits={3} />
              <MetricCell value={block.weight_spearman} digits={3} />
              <MetricCell value={block.sign_agreement} digits={3} />
              <MetricCell value={block.off_rationale_mass} digits={3} />
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  );
}

// -- shared bits ------------------------------------------------------------------

function Section({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="card p-5">
      <div className="mb-3">
        <h2 className="text-base font-semibold text-slate-900">{title}</h2>
        {subtitle && <p className="text-xs text-slate-500">{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}

function MetricCell({
  value,
  digits = 4,
  bold = false,
}: {
  value?: SeedAggregate | number | null;
  digits?: number;
  bold?: boolean;
}) {
  if (value == null) return <td className="py-2 text-right font-mono text-slate-400">—</td>;
  if (typeof value === "number") {
    return (
      <td className={`py-2 text-right font-mono ${bold ? "font-semibold text-slate-900" : ""}`}>
        {fmt(value, digits)}
      </td>
    );
  }
  return (
    <td className={`py-2 text-right font-mono ${bold ? "font-semibold text-slate-900" : ""}`}>
      {fmt(value.mean, digits)}
      {value.std ? <span className="text-slate-400"> ±{fmt(value.std, digits)}</span> : null}
    </td>
  );
}

function CurvePlot({
  title,
  curves,
  xKey,
  yKey,
  xLabel,
  yLabel,
  diagonal = false,
}: {
  title: string;
  curves: { name: string; roc: any; pr: any }[];
  xKey: string;
  yKey: string;
  xLabel: string;
  yLabel: string;
  diagonal?: boolean;
}) {
  const source = xKey === "fpr" ? "roc" : "pr";
  // resample every model's curve onto a shared x-grid so one <LineChart> can hold them all
  const grid = Array.from({ length: 51 }, (_, i) => i / 50);
  const data = grid.map((x) => {
    const row: Record<string, number> = { x };
    for (const curve of curves) {
      const c = (curve as any)[source];
      if (!c) continue;
      row[curve.name] = interpolate(c[xKey], c[yKey], x);
    }
    if (diagonal) row.__chance = x;
    return row;
  });

  return (
    <div>
      <p className="mb-1 text-xs font-medium text-slate-600">{title}</p>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 12, bottom: 22, left: -12 }}>
            <CartesianGrid stroke="#f1f5f9" />
            <XAxis
              dataKey="x"
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickFormatter={(v: number) => v.toFixed(1)}
              label={{ value: xLabel, position: "insideBottom", offset: -12, fontSize: 10, fill: "#64748b" }}
            />
            <YAxis tick={{ fontSize: 10, fill: "#64748b" }} domain={[0, 1]} />
            <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} formatter={(v: number) => v?.toFixed(3)} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            {curves.map((curve, i) => (
              <Line
                key={curve.name}
                type="monotone"
                dataKey={curve.name}
                stroke={CURVE_COLORS[i % CURVE_COLORS.length]}
                dot={false}
                strokeWidth={1.8}
                isAnimationActive={false}
              />
            ))}
            {diagonal && (
              <Line
                dataKey="__chance"
                stroke="#cbd5e1"
                strokeDasharray="4 4"
                dot={false}
                strokeWidth={1}
                name="chance"
                isAnimationActive={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Notice({ title, body }: { title: string; body: string }) {
  return (
    <div className="mx-auto max-w-2xl card p-6 text-center">
      <h2 className="text-base font-semibold text-slate-800">{title}</h2>
      <p className="mt-1.5 text-sm text-slate-500">{body}</p>
    </div>
  );
}

// -- helpers ----------------------------------------------------------------------

function collectCurves(transformers?: any, baselines?: any) {
  const out: { name: string; roc: any; pr: any }[] = [];
  for (const [key, block] of Object.entries<any>(transformers?.models ?? {})) {
    if (block.curves) out.push({ name: key, roc: block.curves.roc, pr: block.curves.pr });
  }
  for (const [key, curves] of Object.entries<any>(baselines?.curves ?? {})) {
    out.push({ name: prettyBaseline(key), roc: curves.roc, pr: curves.pr });
  }
  return out;
}

function buildFaithfulnessCurve(curves: Record<string, any>) {
  const methods = Object.keys(curves);
  if (methods.length === 0) return [];
  const fractions: number[] = curves[methods[0]]?.fractions ?? [];
  return fractions.map((fraction, i) => {
    const row: Record<string, number> = { fraction };
    for (const method of methods) row[method] = curves[method]?.deletion_mean?.[i] ?? NaN;
    return row;
  });
}

function interpolate(xs: number[], ys: number[], x: number): number {
  if (!xs?.length) return NaN;
  // curves may be stored descending (PR) or ascending (ROC)
  const ascending = xs[0] <= xs[xs.length - 1];
  const px = ascending ? xs : [...xs].reverse();
  const py = ascending ? ys : [...ys].reverse();
  if (x <= px[0]) return py[0];
  if (x >= px[px.length - 1]) return py[py.length - 1];
  let lo = 0;
  while (lo < px.length - 1 && px[lo + 1] < x) lo += 1;
  const span = px[lo + 1] - px[lo];
  if (span === 0) return py[lo];
  const t = (x - px[lo]) / span;
  return py[lo] + t * (py[lo + 1] - py[lo]);
}

function prettyBaseline(key: string): string {
  return (
    {
      logreg: "TF-IDF + Logistic Regression",
      random_forest: "TF-IDF + Random Forest",
      xgboost: "TF-IDF + XGBoost",
    }[key] ?? key
  );
}

function fmt(value: number | null | undefined, digits: number): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

function fmtP(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value < 0.001 ? "<0.001" : value.toFixed(3);
}
