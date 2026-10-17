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
import { METHOD_FRIENDLY } from "./PlainLanguage";

const CURVE_COLORS = ["#265f8c", "#c53030", "#0f9d76", "#7c3aed", "#dd6b20", "#0891b2"];

/**
 * Shows the actual evaluation artefacts produced by the scripts in `scripts/`.
 * Every section renders only if its results file exists, so the page degrades to
 * "not run yet" instead of displaying placeholder numbers that could be mistaken
 * for findings.
 *
 * Each section leads with a plain-language reading of its numbers; the raw
 * tables, curves and metric names sit behind a "Technical details" toggle so
 * nothing is lost for a technical reader.
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
        body="The models haven't been tested yet. Once the evaluation pipeline has run, this page will summarise how well they work."
      />
    );
  }

  const ranked = rankModels(data.transformers, data.baselines);

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <OverviewSection report={data.dataset_report} ranked={ranked} />
      {data.dataset_report ? (
        <DatasetSection report={data.dataset_report} />
      ) : (
        <NotEvaluated title="The data" script="scripts/00_build_dataset.py" />
      )}
      {ranked.length > 0 ? (
        <PerformanceSection ranked={ranked} transformers={data.transformers} baselines={data.baselines} />
      ) : (
        <NotEvaluated title="How accurate are the models?" script="scripts/01_train_baselines.py and 02_train_transformers.py" />
      )}
      {data.evaluation?.significance ? (
        <SignificanceSection evaluation={data.evaluation} />
      ) : (
        <NotEvaluated title="Statistical comparison" script="scripts/03_evaluate.py" />
      )}
      {data.faithfulness ? (
        <FaithfulnessSection faithfulness={data.faithfulness} />
      ) : (
        <NotEvaluated title="Explanation faithfulness" script="scripts/04_faithfulness.py" />
      )}
      {data.stability ? (
        <StabilitySection stability={data.stability} />
      ) : (
        <NotEvaluated title="Explanation stability" script="scripts/05_stability.py" />
      )}
      {data.plausibility ? (
        <PlausibilitySection plausibility={data.plausibility} />
      ) : (
        <NotEvaluated title="Explanation plausibility" script="scripts/06_plausibility.py" />
      )}
    </div>
  );
}

// -- overview ---------------------------------------------------------------------

function OverviewSection({ report, ranked }: { report?: any; ranked: RankedModel[] }) {
  if (!report && ranked.length === 0) return null;
  const best = ranked[0];
  const bestAi = ranked.find((r) => r.family === "transformer");
  const outcome = report?.primary_outcome;
  const testRate = report?.splits?.test?.[`${outcome}_rate`];
  const total = ["train", "val", "test"].reduce((sum, s) => sum + (report?.splits?.[s]?.n ?? 0), 0);

  return (
    <section className="card border-clinical-200 bg-gradient-to-br from-clinical-50/60 to-white p-5">
      <h2 className="text-base font-semibold text-slate-900">The short version</h2>
      <ul className="mt-3 space-y-2 text-sm leading-relaxed text-slate-700">
        {report && (
          <li className="flex gap-2">
            <Dot />
            <span>
              The models were trained and tested on <strong>{total.toLocaleString()}</strong>{" "}
              {report.source === "synthetic" ? "computer-generated (synthetic) " : ""}hospital
              discharge notes
              {testRate != null && (
                <>
                  , in which roughly <strong>{oneIn(testRate)}</strong> patients died in hospital
                </>
              )}
              .
            </span>
          </li>
        )}
        {best && (
          <li className="flex gap-2">
            <Dot />
            <span>
              The best performer was <strong>{best.friendly}</strong>. Given one patient who died and
              one who survived, it ranks the right patient as higher-risk about{" "}
              <strong>{tenths(best.agg.auroc?.mean)} times out of 10</strong> ({rating(best.agg.auroc?.mean).label.toLowerCase()}).
            </span>
          </li>
        )}
        {bestAi && best && bestAi !== best && (
          <li className="flex gap-2">
            <Dot />
            <span>
              The best AI language model, <strong>{bestAi.friendly}</strong>, gets it right about{" "}
              <strong>{tenths(bestAi.agg.auroc?.mean)} times out of 10</strong>. That is{" "}
              {(bestAi.agg.auroc?.mean ?? 0) < (best.agg.auroc?.mean ?? 0) ? "behind" : "on par with"}{" "}
              the simpler word-counting approach in this setup, likely because of the small dataset
              and the shortened, laptop-friendly training settings.
            </span>
          </li>
        )}
        <li className="flex gap-2">
          <Dot />
          <span>
            These are research results. No model here has been validated for real clinical use.
          </span>
        </li>
      </ul>
    </section>
  );
}

// -- dataset ----------------------------------------------------------------------

function DatasetSection({ report }: { report: any }) {
  const splits = report.splits ?? {};
  const label = report.primary_outcome;
  const tiles = [
    { key: "train", title: "Learning", body: "notes the models learned from" },
    { key: "val", title: "Tuning", body: "notes used to fine-tune settings such as the alert level" },
    { key: "test", title: "Final test", body: "unseen notes used only to measure the final score" },
  ].filter((t) => splits[t.key]);

  return (
    <Section
      title="The data"
      subtitle={`What the models were trained and tested on${report.source === "synthetic" ? " (synthetic notes, no real patients)" : ""}`}
    >
      <div className="grid gap-3 sm:grid-cols-3">
        {tiles.map((t) => {
          const row = splits[t.key];
          const rate = row[`${label}_rate`];
          return (
            <div key={t.key} className="rounded-lg border border-slate-200 p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{t.title}</p>
              <p className="mt-1 text-2xl font-semibold text-slate-900">{row.n.toLocaleString()}</p>
              <p className="text-xs text-slate-500">{t.body}</p>
              {rate != null && (
                <p className="mt-2 text-xs text-slate-600">
                  About <strong>{oneIn(rate)}</strong> of these patients died in hospital
                </p>
              )}
            </div>
          );
        })}
      </div>
      <p className="mt-3 text-sm leading-relaxed text-slate-600">
        Each patient appears in only one group, so the final test really is on notes the models have
        never seen. Every group has a similar share of deaths, which keeps the comparison fair.
      </p>

      <TechDetails>
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
        <p className="text-xs text-slate-500">
          Source: <span className="font-mono">{report.source}</span> · outcome:{" "}
          <span className="font-mono">{report.primary_outcome}</span>
        </p>
        {report.token_length && (
          <p className="text-xs text-slate-500">
            Note length: median {Math.round(report.token_length.median)} tokens, 95th percentile{" "}
            {Math.round(report.token_length.p95)}, {(report.token_length.frac_over_512 * 100).toFixed(1)}%
            exceed 512 tokens (tokenizer: <span className="font-mono">{report.token_length.tokenizer}</span>).
          </p>
        )}
      </TechDetails>
    </Section>
  );
}

// -- performance ------------------------------------------------------------------

function PerformanceSection({
  ranked,
  transformers,
  baselines,
}: {
  ranked: RankedModel[];
  transformers?: any;
  baselines?: any;
}) {
  const curves = collectCurves(transformers, baselines);
  const byAuprc = [...ranked].sort((a, b) => (b.agg.auprc?.mean ?? 0) - (a.agg.auprc?.mean ?? 0));

  return (
    <Section
      title="How accurate are the models?"
      subtitle="Scored on the final-test notes the models never saw during training"
    >
      <p className="mb-3 text-sm leading-relaxed text-slate-600">
        The score below answers a simple question: <em>given one patient who died and one who
        survived, how often does the model give the higher risk to the right one?</em> A coin flip
        gets 5 out of 10, and a perfect model gets 10 out of 10.
      </p>

      <div className="space-y-2.5">
        {ranked.map((row, index) => {
          const auroc = row.agg.auroc?.mean ?? 0;
          const r = rating(auroc);
          return (
            <div key={row.key} className="rounded-lg border border-slate-200 p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <p className="text-sm font-medium text-slate-800">
                    {index === 0 && <span className="mr-1.5" aria-label="best">🏆</span>}
                    {row.friendly}
                  </p>
                  <p className="text-xs text-slate-500">{row.blurb}</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`chip ${r.className}`}>{r.label}</span>
                  <span className="text-sm font-semibold text-slate-800">{tenths(auroc)} / 10</span>
                </div>
              </div>
              <div className="mt-2 h-2 rounded-full bg-slate-100">
                <div
                  className={`h-2 rounded-full ${r.bar}`}
                  style={{ width: `${Math.max(0, (auroc - 0.5) / 0.5) * 100}%` }}
                />
              </div>
              <div className="mt-1 flex justify-between text-[10px] text-slate-400">
                <span>coin flip</span>
                <span>perfect</span>
              </div>
            </div>
          );
        })}
      </div>

      <TechDetails>
        <p className="text-xs text-slate-500">
          Held-out test set · mean ± std across random seeds. The list above is ranked by AUROC; the
          table below by AUPRC (the project's headline metric).
        </p>
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
              {byAuprc.map((row, index) => (
                <tr
                  key={row.key}
                  className={`border-b border-slate-100 ${index === 0 ? "bg-emerald-50/50" : ""}`}
                >
                  <td className="py-2">
                    <span className="font-medium text-slate-800">{row.name}</span>
                    <span className="ml-2 text-[10px] uppercase tracking-wide text-slate-400">
                      {row.family === "baseline" ? "TF-IDF baseline" : "transformer"}
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

        <Glossary
          items={[
            ["AUROC", "Chance of ranking a random death above a random survivor. 0.5 = chance, 1.0 = perfect."],
            ["AUPRC", "How precise the alerts are across all alert levels. More informative than AUROC when the outcome is rare; chance equals the death rate."],
            ["F1", "Balance of catching deaths (recall) and avoiding false alarms (precision) at the chosen alert level."],
            ["Brier", "Average squared error of the predicted probabilities. Lower is better."],
            ["Seeds", "Number of repeated training runs with different random starts; results are averaged."],
          ]}
        />

        {curves.length > 0 && (
          <div className="grid gap-4 md:grid-cols-2">
            <CurvePlot title="ROC" curves={curves} xKey="fpr" yKey="tpr" xLabel="False positive rate" yLabel="True positive rate" diagonal />
            <CurvePlot title="Precision–Recall" curves={curves} xKey="recall" yKey="precision" xLabel="Recall" yLabel="Precision" />
          </div>
        )}
      </TechDetails>
    </Section>
  );
}

// -- significance -----------------------------------------------------------------

function SignificanceSection({ evaluation }: { evaluation: any }) {
  const sig = evaluation.significance;
  const tests = sig?.tests ?? {};
  const holm = sig?.holm?.results ?? {};
  const entries = Object.entries<any>(tests);

  return (
    <Section
      title="Is the difference real or just luck?"
      subtitle="Comparing the best AI model against each simpler approach"
    >
      <ul className="space-y-2">
        {entries.map(([name, test]) => {
          const diff = test.bootstrap?.difference;
          const significant = holm[name]?.significant;
          const better = (diff ?? 0) > 0;
          return (
            <li key={name} className="flex items-start gap-2 rounded-lg border border-slate-200 p-3 text-sm">
              <span
                className={`chip shrink-0 ${
                  significant
                    ? better
                      ? "bg-emerald-100 text-emerald-800"
                      : "bg-red-100 text-red-800"
                    : "bg-slate-100 text-slate-600"
                }`}
              >
                {significant ? (better ? "Clearly better" : "Clearly worse") : "No clear difference"}
              </span>
              <span className="text-slate-700">
                vs. {friendlyBaseline(name)}
                {!significant && (
                  <span className="text-slate-500"> (the gap is small enough that it could be chance)</span>
                )}
              </span>
            </li>
          );
        })}
      </ul>

      <TechDetails>
        <p className="text-xs text-slate-500">
          Paired bootstrap on {sig?.metric ?? "auprc"} · Holm-adjusted for multiple comparisons.
        </p>
        <div className="overflow-x-auto">
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
              {entries.map(([name, test]) => {
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
        </div>
        <Glossary
          items={[
            ["Δ", "Difference in score between the best model and the comparison model."],
            ["95% CI", "Range the true difference most likely falls in. If it includes 0, the difference may be chance."],
            ["p (Holm)", "Probability of a gap this large by chance, corrected for making several comparisons. Below 0.05 counts as a clear difference."],
          ]}
        />
      </TechDetails>
    </Section>
  );
}

// -- explanation quality ----------------------------------------------------------

function FaithfulnessSection({ faithfulness }: { faithfulness: any }) {
  const methods = faithfulness.methods ?? {};
  const curves = faithfulness.curves ?? {};
  const entries = Object.entries<any>(methods).sort(
    (a, b) => (b[1].comprehensiveness?.mean ?? 0) - (a[1].comprehensiveness?.mean ?? 0),
  );
  const curveData = buildFaithfulnessCurve(curves);

  return (
    <Section
      title="Do the highlights reflect what the model really used?"
      subtitle="Test: hide the highlighted words and see whether the prediction changes"
    >
      <p className="mb-3 text-sm leading-relaxed text-slate-600">
        If an explanation is honest, hiding the words it highlights should change the model&rsquo;s
        answer a lot. Methods are ranked from most to least trustworthy on this test.
      </p>
      <MethodRanking
        entries={entries.map(([m, b]) => [m, b.comprehensiveness?.mean])}
        note="Bigger change when highlighted words are hidden = more faithful"
      />

      <TechDetails>
        <p className="text-xs text-slate-500">
          ERASER comprehensiveness / sufficiency · {faithfulness.n_notes ?? "?"} test notes ·{" "}
          {faithfulness.model ?? ""}
        </p>
        <div className="overflow-x-auto">
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
              {entries.map(([method, block]) => (
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
        </div>
        <Glossary
          items={[
            ["Comprehensiveness", "Drop in predicted risk when the top-highlighted words are removed. Higher is better."],
            ["Sufficiency", "Drop in predicted risk when only the top-highlighted words are kept. Lower is better."],
            ["vs. random", "Comprehensiveness above what randomly chosen words would give."],
            ["Deletion AUC", "Area under the deletion curve below. Lower means confidence falls faster, so the explanation is more faithful."],
          ]}
        />

        {curveData.length > 0 && (
          <div>
            <p className="mb-1 text-xs font-medium text-slate-600">
              Deletion curves: confidence as the highest-scoring words are progressively masked.
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
      </TechDetails>
    </Section>
  );
}

function StabilitySection({ stability }: { stability: any }) {
  const methods = stability.methods ?? {};
  const entries = Object.entries<any>(methods).sort(
    (a, b) => (b[1].paraphrase?.spearman?.mean ?? b[1].paraphrase?.spearman ?? 0) -
      (a[1].paraphrase?.spearman?.mean ?? a[1].paraphrase?.spearman ?? 0),
  );

  return (
    <Section
      title="Are the highlights consistent?"
      subtitle="Test: reword the note without changing its meaning, then compare the highlights"
    >
      <p className="mb-3 text-sm leading-relaxed text-slate-600">
        A reliable explanation should highlight the same things when a note says the same thing in
        different words. Methods are ranked from most to least consistent.
      </p>
      <MethodRanking
        entries={entries.map(([m, b]) => [m, valueOf(b.paraphrase?.spearman)])}
        note="Higher = highlights stay the same after rewording"
      />

      <TechDetails>
        <p className="text-xs text-slate-500">
          Agreement under meaning-preserving edits · {stability.n_notes ?? "?"} notes
        </p>
        <div className="overflow-x-auto">
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
              {entries.map(([method, block]) => (
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
        </div>
        <Glossary
          items={[
            ["Paraphrase ρ", "Rank correlation of word importances before and after rewording (1 = identical)."],
            ["Top-10 Jaccard", "Overlap of the ten most important words before and after rewording."],
            ["Near-dup ρ", "Same as paraphrase ρ, for tiny edits such as punctuation changes."],
            ["Distractor mass", "Share of importance landing on an irrelevant sentence that was inserted. Lower is better."],
          ]}
        />
      </TechDetails>
    </Section>
  );
}

function PlausibilitySection({ plausibility }: { plausibility: any }) {
  const methods = plausibility.methods ?? {};
  const entries = Object.entries<any>(methods).sort(
    (a, b) => (valueOf(b[1].token_f1) ?? 0) - (valueOf(a[1].token_f1) ?? 0),
  );

  return (
    <Section
      title="Do the highlights match the real risk factors?"
      subtitle="Test: compare highlighted words with the risk factors known to be in each note"
    >
      <p className="mb-3 text-sm leading-relaxed text-slate-600">
        Each note contains known risk factors (for example sepsis or low blood pressure). A good
        explanation should point at those, not at unrelated words. Methods are ranked from best to
        worst match.
      </p>
      <MethodRanking
        entries={entries.map(([m, b]) => [m, valueOf(b.token_f1)])}
        note="Higher = highlights line up better with the true risk factors"
      />

      <TechDetails>
        <p className="text-xs text-slate-500">
          Agreement with the gold rationale spans recorded by the corpus generator
        </p>
        <div className="overflow-x-auto">
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
              {entries.map(([method, block]) => (
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
        </div>
        <Glossary
          items={[
            ["Token F1", "Overlap between the top-highlighted words and the true risk-factor words."],
            ["Rationale AUPRC", "How well importance scores separate risk-factor words from the rest."],
            ["Weight ρ", "Correlation between importance and each risk factor's true strength."],
            ["Sign agreement", "How often the method gets the direction (raises vs. lowers risk) right."],
            ["Off-rationale", "Share of importance on words that are not risk factors. Lower is better."],
          ]}
        />
      </TechDetails>
    </Section>
  );
}

/** Ordered list of explanation methods with a relative bar, used by the three explanation sections. */
function MethodRanking({ entries, note }: { entries: [string, number | null | undefined][]; note: string }) {
  const values = entries.map(([, v]) => v ?? 0);
  const max = Math.max(...values.map(Math.abs), 1e-9);
  return (
    <div>
      <ol className="space-y-2">
        {entries.map(([method, value], index) => (
          <li key={method} className="flex items-center gap-3">
            <span className="w-5 text-right text-xs font-semibold text-slate-400">{index + 1}.</span>
            <span className="w-56 shrink-0 text-sm text-slate-700">
              {METHOD_FRIENDLY[method]?.name ?? method}
            </span>
            <div className="h-2 flex-1 rounded-full bg-slate-100">
              <div
                className={`h-2 rounded-full ${index === 0 ? "bg-emerald-500" : "bg-clinical-500/70"}`}
                style={{ width: `${(Math.max(value ?? 0, 0) / max) * 100}%` }}
              />
            </div>
          </li>
        ))}
      </ol>
      <p className="mt-2 text-[11px] text-slate-400">{note}</p>
    </div>
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

function TechDetails({ children }: { children: React.ReactNode }) {
  return (
    <details className="mt-4 rounded-lg border border-slate-200">
      <summary className="cursor-pointer select-none px-3 py-2 text-xs font-medium text-slate-500 hover:text-slate-700">
        Technical details
      </summary>
      <div className="space-y-4 border-t border-slate-100 p-3">{children}</div>
    </details>
  );
}

function Glossary({ items }: { items: [string, string][] }) {
  return (
    <dl className="grid gap-x-4 gap-y-1.5 rounded-lg bg-slate-50 p-3 text-xs sm:grid-cols-[max-content_1fr]">
      {items.map(([term, def]) => (
        <div key={term} className="contents">
          <dt className="font-mono font-semibold text-slate-700">{term}</dt>
          <dd className="text-slate-600">{def}</dd>
        </div>
      ))}
    </dl>
  );
}

function Dot() {
  return <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-clinical-500" aria-hidden />;
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

function NotEvaluated({ title, script }: { title: string; script: string }) {
  return (
    <section className="card flex flex-wrap items-center justify-between gap-2 p-5">
      <div>
        <h2 className="text-base font-semibold text-slate-900">{title}</h2>
        <p className="text-xs text-slate-500">
          No results file yet. Produced by <span className="font-mono">{script}</span>.
        </p>
      </div>
      <span className="chip bg-slate-100 text-slate-600">Not evaluated yet</span>
    </section>
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

interface RankedModel {
  key: string;
  name: string;
  friendly: string;
  blurb: string;
  family: "baseline" | "transformer";
  agg: Record<string, SeedAggregate>;
}

const MODEL_FRIENDLY: Record<string, { friendly: string; blurb: string }> = {
  logreg: { friendly: "Word-count model (logistic regression)", blurb: "Simple approach: weighs how often key words appear" },
  random_forest: { friendly: "Word-count model (random forest)", blurb: "Simple approach: many small decision trees over word counts" },
  xgboost: { friendly: "Word-count model (XGBoost)", blurb: "Simple approach: boosted decision trees over word counts" },
  clinicalbert: { friendly: "ClinicalBERT", blurb: "AI language model pre-trained on hospital notes; reads about 75 words" },
  clinicalbert_512: {
    friendly: "ClinicalBERT (full note)",
    blurb: "Same model retrained to read about 300 words, the whole of almost every note",
  },
  biobert: { friendly: "BioBERT", blurb: "AI language model pre-trained on biomedical research papers" },
  pubmedbert: { friendly: "PubMedBERT", blurb: "AI language model pre-trained on PubMed medical abstracts" },
  bert: { friendly: "BERT (general)", blurb: "General-purpose AI language model, not medical-specific" },
};

/** All models with aggregate scores, best first by AUROC (the metric the plain-language view explains). */
function rankModels(transformers?: any, baselines?: any): RankedModel[] {
  const rows: RankedModel[] = [];
  for (const [key, block] of Object.entries<any>(baselines?.models ?? {})) {
    if (block.aggregate)
      rows.push({ key, name: prettyBaseline(key), family: "baseline", agg: block.aggregate, ...friendlyModel(key) });
  }
  for (const [key, block] of Object.entries<any>(transformers?.models ?? {})) {
    if (block.aggregate)
      rows.push({ key, name: key, family: "transformer", agg: block.aggregate, ...friendlyModel(key) });
  }
  return rows.sort((a, b) => (b.agg.auroc?.mean ?? 0) - (a.agg.auroc?.mean ?? 0));
}

function friendlyModel(key: string) {
  return MODEL_FRIENDLY[key] ?? { friendly: key, blurb: "" };
}

function rating(auroc?: number) {
  const v = auroc ?? 0;
  if (v >= 0.85) return { label: "Good", className: "bg-emerald-100 text-emerald-800", bar: "bg-emerald-500" };
  if (v >= 0.75) return { label: "Fair", className: "bg-sky-100 text-sky-800", bar: "bg-sky-500" };
  if (v >= 0.65) return { label: "Modest", className: "bg-amber-100 text-amber-800", bar: "bg-amber-500" };
  return { label: "Weak", className: "bg-red-100 text-red-800", bar: "bg-red-400" };
}

function tenths(x?: number) {
  return x == null ? "—" : (x * 10).toFixed(1).replace(/\.0$/, "");
}

/** "1 in 6" style frequency for a rate. */
function oneIn(rate: number) {
  if (!rate || rate <= 0) return "none";
  return `1 in ${Math.round(1 / rate)}`;
}

function valueOf(v: SeedAggregate | number | null | undefined): number | undefined {
  if (v == null) return undefined;
  return typeof v === "number" ? v : v.mean;
}

function friendlyBaseline(key: string) {
  return MODEL_FRIENDLY[key]?.friendly ?? prettyBaseline(key);
}

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
