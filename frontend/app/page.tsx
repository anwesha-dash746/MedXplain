"use client";

import { useCallback, useEffect, useState } from "react";
import { Controls } from "@/components/Controls";
import { HistoryPanel } from "@/components/HistoryPanel";
import { NoteInput } from "@/components/NoteInput";
import { ResultPanel } from "@/components/ResultPanel";
import { ResultsView } from "@/components/ResultsView";
import {
  ApiError,
  getExamples,
  getHealth,
  getMethods,
  getModels,
  predict,
} from "@/lib/api";
import type {
  ExampleNote,
  ExplanationMethod,
  HealthResponse,
  MethodInfo,
  ModelInfo,
  PredictResponse,
} from "@/lib/types";

const MAX_CHARS = 40000;
type Tab = "analyse" | "results" | "about";

export default function Dashboard() {
  const [tab, setTab] = useState<Tab>("analyse");

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [methods, setMethods] = useState<MethodInfo[]>([]);
  const [examples, setExamples] = useState<ExampleNote[]>([]);

  const [note, setNote] = useState("");
  const [analysedNote, setAnalysedNote] = useState("");
  const [model, setModel] = useState("");
  const [method, setMethod] = useState<ExplanationMethod>("attention_rollout");
  const [compareMode, setCompareMode] = useState(false);
  const [compareMethods, setCompareMethods] = useState<ExplanationMethod[]>([
    "attention_rollout",
    "shap",
  ]);

  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<{ message: string; hint?: string } | null>(null);
  const [historyVersion, setHistoryVersion] = useState(0);

  // Bootstrap: everything the controls need comes from the API, so the UI can
  // never offer a model or method the server cannot actually serve.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [h, m, me, ex] = await Promise.all([
          getHealth(),
          getModels(),
          getMethods(),
          getExamples().catch(() => [] as ExampleNote[]),
        ]);
        if (cancelled) return;
        setHealth(h);
        setModels(m);
        setMethods(me);
        setExamples(ex);
        const firstAvailable = m.find((x) => x.available);
        setModel(firstAvailable?.key ?? h.default_model);
        setMethod((h.default_method as ExplanationMethod) ?? "attention_rollout");
      } catch (err) {
        if (!cancelled) setError(toError(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const runPrediction = useCallback(async () => {
    if (!note.trim()) return;
    setLoading(true);
    setError(null);
    const submitted = note;
    try {
      const response = await predict({
        text: submitted,
        model: model || undefined,
        method: compareMode ? undefined : method,
        compareMethods: compareMode ? compareMethods : undefined,
      });
      setResult(response);
      // The backend cleans the note before scoring; highlight offsets refer to
      // the submitted text, so keep the exact string that was analysed.
      setAnalysedNote(submitted);
      setHistoryVersion((v) => v + 1);
    } catch (err) {
      setError(toError(err));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [note, model, method, compareMode, compareMethods]);

  const noModels = models.length > 0 && !models.some((m) => m.available);

  return (
    <div className="min-h-screen">
      <Header health={health} tab={tab} onTabChange={setTab} />

      <main className="mx-auto max-w-[1400px] px-4 py-6">
        {error && (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-4">
            <p className="text-sm font-medium text-red-800">{error.message}</p>
            {error.hint && (
              <p className="mt-1 font-mono text-xs text-red-600">{error.hint}</p>
            )}
          </div>
        )}

        {noModels && (
          <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
            <p className="font-medium">No trained checkpoint is available yet.</p>
            <p className="mt-1 font-mono text-xs">python scripts/02_train_transformers.py</p>
          </div>
        )}

        {tab === "analyse" && (
          <div className="grid gap-4 xl:grid-cols-[minmax(380px,1fr)_minmax(0,1.9fr)]">
            <div className="space-y-4">
              <NoteInput
                value={note}
                onChange={setNote}
                onSubmit={runPrediction}
                loading={loading}
                examples={examples}
                maxChars={MAX_CHARS}
              />
              <Controls
                models={models}
                methods={methods}
                selectedModel={model}
                selectedMethod={method}
                compareMode={compareMode}
                compareMethods={compareMethods}
                onModelChange={setModel}
                onMethodChange={setMethod}
                onCompareToggle={setCompareMode}
                onCompareMethodsChange={setCompareMethods}
                disabled={loading}
              />
              <HistoryPanel version={historyVersion} />
            </div>

            <div>
              {loading && <LoadingState compareMode={compareMode} methods={compareMethods} />}
              {!loading && result && <ResultPanel result={result} noteText={analysedNote} />}
              {!loading && !result && <EmptyState />}
            </div>
          </div>
        )}

        {tab === "results" && <ResultsView />}
        {tab === "about" && <About />}
      </main>

      <footer className="mx-auto max-w-[1400px] px-4 pb-8 pt-2 text-xs text-slate-400">
        MedXplain — research prototype. Not a medical device; not for clinical use.
      </footer>
    </div>
  );
}

function Header({
  health,
  tab,
  onTabChange,
}: {
  health: HealthResponse | null;
  tab: Tab;
  onTabChange: (tab: Tab) => void;
}) {
  const tabs: { key: Tab; label: string }[] = [
    { key: "analyse", label: "Analyse" },
    { key: "results", label: "Results" },
    { key: "about", label: "About" },
  ];
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-4 px-4 py-3">
        <div className="flex items-center gap-2.5">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-clinical-600 text-sm font-bold text-white">
            Mx
          </div>
          <div>
            <h1 className="text-base font-semibold leading-tight text-slate-900">MedXplain</h1>
            <p className="text-[11px] leading-tight text-slate-500">
              Explainable clinical risk prediction
            </p>
          </div>
        </div>

        <nav className="flex gap-1 rounded-lg bg-slate-100 p-1">
          {tabs.map((t) => (
            <button
              key={t.key}
              type="button"
              onClick={() => onTabChange(t.key)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                tab === t.key ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-2 text-xs text-slate-500">
          <span
            className={`h-2 w-2 rounded-full ${health ? "bg-emerald-500" : "bg-slate-300"}`}
            aria-hidden
          />
          {health ? (
            <span>
              API v{health.version} · {health.device} ·{" "}
              {health.models_available.length} model
              {health.models_available.length === 1 ? "" : "s"}
            </span>
          ) : (
            <span>connecting…</span>
          )}
        </div>
      </div>
    </header>
  );
}

function LoadingState({ compareMode, methods }: { compareMode: boolean; methods: string[] }) {
  return (
    <div className="card p-6">
      <div className="space-y-3">
        <div className="skeleton h-6 w-48" />
        <div className="skeleton h-28 w-full" />
        <div className="skeleton h-4 w-full" />
        <div className="skeleton h-4 w-5/6" />
        <div className="skeleton h-4 w-4/6" />
      </div>
      <p className="mt-4 text-xs text-slate-500">
        Running transformer inference
        {compareMode ? ` and ${methods.length} explanation methods` : " and the explanation"}.
        Perturbation-based methods (SHAP, occlusion) need hundreds of model evaluations and can
        take tens of seconds on CPU.
      </p>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="card grid min-h-[420px] place-items-center p-8 text-center">
      <div className="max-w-md">
        <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-full bg-clinical-50 text-clinical-600">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M4 5h11l5 5v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1Z" />
            <path d="M15 5v5h5M7 13h8M7 17h5" />
          </svg>
        </div>
        <h2 className="text-base font-semibold text-slate-800">No note analysed yet</h2>
        <p className="mt-1.5 text-sm leading-relaxed text-slate-500">
          Paste a discharge summary on the left, or load one of the case-study notes, then press
          <span className="mx-1 font-medium text-slate-700">Analyse note</span>
          to get a risk score with the words that drove it.
        </p>
      </div>
    </div>
  );
}

function About() {
  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <section className="card p-6">
        <h2 className="text-lg font-semibold text-slate-900">What this is</h2>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          MedXplain predicts a clinical risk outcome from the free text of a clinical note using a
          fine-tuned domain-adapted transformer, and explains the prediction at the level of
          individual words. It exists to test a specific claim: that an explanation can be made
          faithful enough — not merely plausible-looking — to be worth putting in front of a
          clinician.
        </p>
      </section>

      <section className="card p-6">
        <h2 className="text-lg font-semibold text-slate-900">Reading the explanation honestly</h2>
        <ul className="mt-2 space-y-2 text-sm leading-relaxed text-slate-600">
          <li>
            <strong className="text-slate-800">Attention is not attribution.</strong> Attention
            rollout shows which words fed the classifier&rsquo;s summary representation. It has no
            sign, so it cannot tell you whether a word raised or lowered the risk. SHAP,
            integrated gradients and occlusion can.
          </li>
          <li>
            <strong className="text-slate-800">The threshold is not 50%.</strong> For a rare
            outcome the operating point is chosen on a validation split to balance recall against
            false alarms, and it is usually well below one half.
          </li>
          <li>
            <strong className="text-slate-800">Truncation is visible.</strong> If a note exceeds the
            model&rsquo;s context window the response says so, because the highlights then describe
            only the part the model read.
          </li>
        </ul>
      </section>

      <section className="card p-6">
        <h2 className="text-lg font-semibold text-slate-900">Limitations</h2>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          This is a research prototype and not a medical device. The shipped model is trained on a
          synthetic corpus that imitates the structure of MIMIC-III discharge summaries so that the
          repository is runnable without credentialed data access; numbers from that corpus
          characterise the pipeline, not clinical performance. Results on a real cohort require
          PhysioNet approval and are produced by pointing the same configuration at MIMIC-III.
        </p>
      </section>
    </div>
  );
}

function toError(err: unknown): { message: string; hint?: string } {
  if (err instanceof ApiError) return { message: err.message, hint: err.hint };
  return { message: err instanceof Error ? err.message : "Unexpected error" };
}
