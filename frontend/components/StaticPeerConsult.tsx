"use client";

import { useEffect, useState } from "react";
import {
  MAX_QUESTIONS,
  PHYSICIANS,
  GENERATED_NOTE,
  RESPONSE_SETS,
  buildThread,
  type PeerThread,
} from "@/lib/staticPeer";

/**
 * Peer Consult. Questions get five fixed replies chosen by keyword matching in
 * lib/staticPeer.ts. Nothing is sent anywhere:
 * the note, the questions and the replies live only in this browser tab
 * (sessionStorage, so a reload keeps the session and its 2-question limit).
 */

const STORAGE_KEY = "medxplain.peerConsult.threads";

function loadThreads(): PeerThread[] {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed)
      ? parsed.filter((t) => t && typeof t.question === "string" && (t.set === "diagnosis" || t.set === "discharge")).slice(0, MAX_QUESTIONS)
      : [];
  } catch {
    return [];
  }
}

function saveThreads(threads: PeerThread[]) {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
  } catch {
    /* storage unavailable: the session lasts until the page is closed */
  }
}

export function StaticPeerConsult({
  latestNote,
  prefillSignal,
}: {
  /** The most recently analysed note, if any. */
  latestNote: string;
  /** Changes when Results' "Ask a Peer Physician" is pressed, forcing a fresh prefill. */
  prefillSignal: number;
}) {
  const [note, setNote] = useState(latestNote);
  const [noteEdited, setNoteEdited] = useState(false);
  const [question, setQuestion] = useState("");
  const [threads, setThreads] = useState<PeerThread[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setThreads(loadThreads()), []);

  // follow the latest analysed note until the user starts editing it
  useEffect(() => {
    if (!noteEdited && latestNote) setNote(latestNote);
  }, [latestNote, noteEdited]);

  // an explicit "Ask a Peer Physician" always loads the latest analysed note
  useEffect(() => {
    if (prefillSignal > 0 && latestNote) {
      setNote(latestNote);
      setNoteEdited(false);
      setError(null);
    }
  }, [prefillSignal, latestNote]);

  const used = threads.length;
  const limitReached = used >= MAX_QUESTIONS;
  const prefilled = Boolean(latestNote) && !noteEdited && note === latestNote;

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const result = buildThread(note, question, threads);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    const next = [...threads, result.thread];
    setThreads(next);
    saveThreads(next);
    setQuestion("");
    setError(null);
  }

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <section className="card p-5">
        <h2 className="text-xl font-semibold text-slate-900">Peer Consult</h2>
        <p className="text-sm text-slate-600">Explore additional clinical perspectives on the current case.</p>
      </section>

      <section className="card p-5">
        <form onSubmit={submit} className="space-y-3" noValidate>
          <label className="block">
            <span className="text-sm font-medium text-slate-700">Discharge note</span>
            <span className="mt-0.5 block text-xs text-slate-500">
              {prefilled
                ? "Prefilled from the latest analysed note. You can edit it."
                : latestNote
                  ? "Edited. Use “Reload analysed note” to start from the analysed note again."
                  : "No note has been analysed yet. Paste or type the discharge note here."}
            </span>
            <textarea
              value={note}
              onChange={(e) => {
                setNote(e.target.value);
                setNoteEdited(true);
              }}
              disabled={limitReached}
              rows={9}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono text-[13px] text-slate-800 focus:border-clinical-500 focus:outline-none focus:ring-2 focus:ring-clinical-100 disabled:bg-slate-50"
              placeholder="Paste the discharge note…"
            />
          </label>
          {latestNote && !prefilled && !limitReached && (
            <button
              type="button"
              onClick={() => {
                setNote(latestNote);
                setNoteEdited(false);
              }}
              className="text-xs text-clinical-600 hover:underline"
            >
              Reload analysed note
            </button>
          )}

          <label className="block">
            <span className="text-sm font-medium text-slate-700">Clinical question</span>
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={limitReached}
              maxLength={500}
              placeholder="e.g. What could explain the deterioration on day 3?"
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus:border-clinical-500 focus:outline-none focus:ring-2 focus:ring-clinical-100 disabled:bg-slate-50"
            />
          </label>

          {error && (
            <p role="alert" className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
              {error}
            </p>
          )}

          <div className="flex flex-wrap items-center justify-between gap-3">
            <span
              className={`chip ${limitReached ? "bg-slate-200 text-slate-700" : "bg-clinical-50 text-clinical-700"}`}
              aria-live="polite"
            >
              Questions used: {used}/{MAX_QUESTIONS}
            </span>
            <button
              type="submit"
              disabled={limitReached}
              className="rounded-lg bg-clinical-600 px-4 py-2 text-sm font-medium text-white hover:bg-clinical-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Submit Question
            </button>
          </div>
          {limitReached && (
            <p className="text-xs text-slate-500">
              You have used both questions for this session. Your earlier questions and responses stay below.
            </p>
          )}
          <p className="text-xs text-slate-400">
            Your note and questions stay in this browser tab. Nothing is sent to a server.
          </p>
        </form>
      </section>

      <section className="space-y-4" aria-labelledby="peer-responses">
        <h2 id="peer-responses" className="px-1 text-base font-semibold text-slate-900">
          Peer Responses
        </h2>
        {threads.length === 0 ? (
          <div className="card p-6 text-center text-sm text-slate-500">
            No questions yet. Submit a question above to see responses from five physicians.
          </div>
        ) : (
          threads.map((t) => <Thread key={t.id} thread={t} />)
        )}
      </section>
    </div>
  );
}

function Thread({ thread }: { thread: PeerThread }) {
  const set = RESPONSE_SETS[thread.set];
  return (
    <article className="card p-5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="chip bg-clinical-50 text-clinical-700">Question {thread.id}</span>
        <span className="text-xs text-slate-500">{set.title}</span>
        <span className="text-xs text-slate-400">· {new Date(thread.askedAt).toLocaleTimeString()}</span>
      </div>
      <p className="mt-2 text-sm font-medium text-slate-900">{thread.question}</p>
      <details className="mt-2 rounded-lg border border-slate-200">
        <summary className="cursor-pointer select-none px-3 py-1.5 text-xs text-slate-500 hover:text-slate-700">
          Submitted note ({thread.note.length.toLocaleString()} characters)
        </summary>
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap border-t border-slate-100 p-3 font-mono text-[12px] text-slate-600">
          {thread.note}
        </pre>
      </details>

      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {PHYSICIANS.map((p, i) => (
          <div key={p.id} className="rounded-lg border border-slate-200 bg-white p-4">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{p.label}</p>
            <p className="text-sm font-semibold text-slate-900">{p.name}</p>
            <p className="text-xs text-slate-500">
              {p.specialty} · {p.years} years&rsquo; experience
            </p>
            <p className="mt-2 text-sm leading-relaxed text-slate-700">{set.responses[i]}</p>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[11px] text-slate-400">
        {GENERATED_NOTE} Not patient-specific medical advice.
      </p>
    </article>
  );
}
