"use client";

import { useCallback, useEffect, useState } from "react";
import { ApiError } from "@/lib/api";
import {
  findIdentifiers,
  peerApi,
  type Box,
  type ConsultDetail,
  type ConsultSummary,
  type HistoryEvent,
  type Message,
  type Page,
} from "@/lib/peer";
import {
  Empty,
  ErrorBox,
  Pager,
  StatusChip,
  formatTime,
  inputClass,
  primaryButton,
  secondaryButton,
} from "./ui";
import { ExampleResponse } from "./ExampleResponse";

const BOX_TEXT: Record<Box, { title: string; empty: string }> = {
  sent: { title: "My Requests", empty: "You have not sent any consultation requests." },
  received: { title: "Requests Received", empty: "No requests are waiting for you." },
  active: { title: "In Discussion", empty: "No consultations are in progress." },
  completed: { title: "Completed", empty: "No closed, declined or cancelled consultations yet." },
};

function errorText(err: unknown, fallback: string) {
  return err instanceof ApiError ? err.message : fallback;
}

export function ConsultationList({
  token,
  box,
  onOpen,
}: {
  token: string;
  box: Box;
  onOpen: (id: string) => void;
}) {
  const [page, setPage] = useState(1);
  const [data, setData] = useState<Page<ConsultSummary> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => setPage(1), [box]);
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    peerApi
      .consultations(token, box, page)
      .then((d) => !cancelled && setData(d))
      .catch((err) => !cancelled && setError(errorText(err, "Could not load consultations.")))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [token, box, page]);

  return (
    <section className="card p-5">
      <h2 className="mb-3 text-base font-semibold text-slate-900">{BOX_TEXT[box].title}</h2>
      <ErrorBox message={error} />
      {loading && <div className="skeleton h-20 w-full" />}
      {!loading && data && data.items.length === 0 && <Empty title={BOX_TEXT[box].empty} />}
      {!loading && data && data.items.length > 0 && (
        <ul className="divide-y divide-slate-100 rounded-lg border border-slate-200">
          {data.items.map((c) => (
            <li key={c.id}>
              <button
                type="button"
                onClick={() => onOpen(c.id)}
                className="flex w-full flex-wrap items-center gap-x-3 gap-y-1 px-4 py-3 text-left hover:bg-slate-50"
              >
                <span className="min-w-0 flex-1 basis-60">
                  <span className="block truncate text-sm font-medium text-slate-900">{c.title}</span>
                  <span className="block text-xs text-slate-500">
                    {c.consult_type_label} · {c.specialty}
                    {c.counterpart_name
                      ? ` · ${c.role === "requester" ? "with" : "from"} ${c.counterpart_name}`
                      : c.open_request
                        ? " · open request"
                        : ""}
                  </span>
                </span>
                {c.priority === "time_sensitive" && <span className="chip bg-amber-100 text-amber-900">Time-sensitive</span>}
                <StatusChip status={c.status} label={c.status_label} />
                <span className="text-xs text-slate-400">{formatTime(c.updated_at)}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {data && <Pager page={data.page} pageSize={data.page_size} total={data.total} onPage={setPage} />}
    </section>
  );
}

// -----------------------------------------------------------------------------------

const FLOW: { key: string; label: string }[] = [
  { key: "requested", label: "Requested" },
  { key: "accepted", label: "Accepted" },
  { key: "in_discussion", label: "In Discussion" },
  { key: "response_received", label: "Response Received" },
  { key: "closed", label: "Closed" },
];

function Progress({ status }: { status: string }) {
  if (status === "declined" || status === "cancelled") {
    return (
      <p className="text-xs text-slate-500">
        This consultation was {status}. No further steps are possible.
      </p>
    );
  }
  const current = FLOW.findIndex((s) => s.key === status);
  return (
    <ol className="flex flex-wrap items-center gap-1 text-xs" aria-label="Consultation progress">
      {FLOW.map((s, i) => (
        <li key={s.key} className="flex items-center gap-1">
          <span
            aria-current={i === current ? "step" : undefined}
            className={`rounded-full px-2 py-0.5 ${
              i < current
                ? "bg-clinical-100 text-clinical-800"
                : i === current
                  ? "bg-clinical-600 text-white"
                  : "bg-slate-100 text-slate-400"
            }`}
          >
            {s.label}
          </span>
          {i < FLOW.length - 1 && <span className="text-slate-300">→</span>}
        </li>
      ))}
    </ol>
  );
}

export function ConsultationDetail({
  token,
  id,
  onBack,
  onChanged,
}: {
  token: string;
  id: string;
  onBack: () => void;
  onChanged: () => void;
}) {
  const [c, setC] = useState<ConsultDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<string | null>(null);
  const [reason, setReason] = useState("");
  const [history, setHistory] = useState<HistoryEvent[] | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      setC(await peerApi.consultation(token, id));
    } catch (err) {
      setError(errorText(err, "Could not load this consultation."));
    }
  }, [token, id]);

  useEffect(() => {
    load();
  }, [load]);

  async function act(action: string) {
    setBusy(true);
    setActionError(null);
    try {
      setC(await peerApi.act(token, id, action, action === "decline" ? reason.trim() || undefined : undefined));
      setPending(null);
      setReason("");
      setHistory(null);
      onChanged();
    } catch (err) {
      setActionError(errorText(err, "The action failed."));
    } finally {
      setBusy(false);
    }
  }

  async function showHistory() {
    try {
      setHistory(await peerApi.history(token, id));
    } catch (err) {
      setActionError(errorText(err, "Could not load the history."));
    }
  }

  if (error) {
    return (
      <section className="card space-y-3 p-5">
        <ErrorBox message={error} />
        <button type="button" className={secondaryButton} onClick={onBack}>
          Back
        </button>
      </section>
    );
  }
  if (!c) return <div className="card p-5"><div className="skeleton h-40 w-full" /></div>;

  const can = (a: string) => c.allowed_actions.includes(a);
  const ACTION_LABEL: Record<string, string> = {
    accept: "Accept request",
    decline: "Decline",
    cancel: "Cancel request",
    close: "Close consultation",
  };

  return (
    <section className="card space-y-4 p-5">
      <button type="button" onClick={onBack} className="text-xs text-clinical-600 hover:underline">
        ← Back
      </button>
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">{c.title}</h2>
          <p className="text-xs text-slate-500">
            {c.consult_type_label} · {c.specialty}
            {c.subspecialty ? ` · ${c.subspecialty}` : ""} · {c.priority_label} · ID{" "}
            <span className="font-mono">{c.id.slice(0, 8)}</span>
          </p>
        </div>
        <StatusChip status={c.status} label={c.status_label} />
      </div>
      <Progress status={c.status} />

      <dl className="grid gap-x-4 gap-y-1 text-sm sm:grid-cols-[max-content_1fr]">
        <dt className="text-slate-500">Requested by</dt>
        <dd className="text-slate-800">{c.requester.name ?? "—"}</dd>
        <dt className="text-slate-500">Consulting clinician</dt>
        <dd className="text-slate-800">{c.assigned?.name ?? "Not yet accepted (open request)"}</dd>
        <dt className="text-slate-500">Created</dt>
        <dd className="text-slate-800">{formatTime(c.created_at)}</dd>
        {c.closed_at && (
          <>
            <dt className="text-slate-500">Finished</dt>
            <dd className="text-slate-800">{formatTime(c.closed_at)}</dd>
          </>
        )}
      </dl>

      <div className="grid gap-3 md:grid-cols-2">
        <Block title="De-identified case summary" text={c.summary} />
        <Block title="Symptoms and findings" text={c.findings} />
        <Block title="Documented diagnosis or differential" text={c.diagnosis} />
        <Block title="Clinical question" text={c.question} />
        {c.matching_preferences && <Block title="Matching preferences" text={c.matching_preferences} />}
        {c.decline_reason && <Block title="Reason for declining" text={c.decline_reason} />}
      </div>

      <p className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
        Colleagues&rsquo; opinions are advisory and based only on the information shared. They are not
        a diagnosis and do not change any treatment plan. The treating clinician remains responsible
        for patient care.
      </p>

      {c.role === "requester" && <ExampleResponse consultType={c.consult_type} />}

      {c.allowed_actions.some((a) => a in ACTION_LABEL) && (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            {(["accept", "decline", "cancel", "close"] as const).filter(can).map((a) => (
              <button
                key={a}
                type="button"
                disabled={busy}
                onClick={() => (a === "accept" ? act(a) : setPending(a))}
                className={a === "accept" ? primaryButton : secondaryButton}
              >
                {ACTION_LABEL[a]}
              </button>
            ))}
          </div>
          {pending && (
            <div className="space-y-2 rounded-lg border border-slate-300 p-3">
              <p className="text-sm text-slate-800">
                {pending === "decline" && "Decline this request? The requester will see your reason."}
                {pending === "cancel" && "Cancel this request? This cannot be undone."}
                {pending === "close" && "Close this consultation? No further messages can be added."}
              </p>
              {pending === "decline" && (
                <textarea
                  className={`${inputClass} min-h-16`}
                  placeholder="Reason (optional, no identifying details)"
                  value={reason}
                  maxLength={1000}
                  onChange={(e) => setReason(e.target.value)}
                />
              )}
              <div className="flex gap-2">
                <button type="button" className={primaryButton} disabled={busy} onClick={() => act(pending)}>
                  Yes, {pending}
                </button>
                <button type="button" className={secondaryButton} disabled={busy} onClick={() => setPending(null)}>
                  Keep it
                </button>
              </div>
            </div>
          )}
        </div>
      )}
      <ErrorBox message={actionError} />

      <div>
        <h3 className="mb-2 text-sm font-semibold text-slate-900">Discussion</h3>
        {c.messages.length === 0 ? (
          <p className="text-sm italic text-slate-400">
            {c.status === "requested" ? "Messages open once the request is accepted." : "No messages yet."}
          </p>
        ) : (
          <ul className="space-y-2">
            {c.messages.map((m) => (
              <MessageItem key={m.id} m={m} />
            ))}
          </ul>
        )}
      </div>

      {(can("message") || can("opinion")) && (
        <Composer token={token} id={id} allowOpinion={can("opinion")} onPosted={(d) => { setC(d); setHistory(null); onChanged(); }} />
      )}

      <div>
        {history === null ? (
          <button type="button" className="text-xs text-clinical-600 hover:underline" onClick={showHistory}>
            Show status history
          </button>
        ) : (
          <div className="rounded-lg border border-slate-200 p-3">
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">History</p>
            <ul className="space-y-0.5 text-xs text-slate-600">
              {history.map((h, i) => (
                <li key={i}>
                  {formatTime(h.created_at)} · {h.actor_name ?? "Unknown"} · {HISTORY_LABEL[h.action] ?? h.action}
                  {h.to_status ? ` (now ${h.to_status.replace(/_/g, " ")})` : ""}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}

const HISTORY_LABEL: Record<string, string> = {
  created: "sent the request",
  accept: "accepted",
  decline: "declined",
  cancel: "cancelled the request",
  close: "closed the consultation",
  message_posted: "posted a message",
  opinion_posted: "posted an opinion",
};

function Block({ title, text }: { title: string; text: string | null }) {
  return (
    <div className="rounded-lg border border-slate-200 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</p>
      <p className="mt-1 whitespace-pre-wrap text-sm text-slate-800">{text || <span className="italic text-slate-400">Not provided</span>}</p>
    </div>
  );
}

const OPINION_LABELS: Record<string, string> = {
  interpretation: "Interpretation",
  considerations: "Relevant considerations",
  questions: "Questions",
  limitations: "Limitations",
  references: "References",
};

function MessageItem({ m }: { m: Message }) {
  const opinion = m.kind === "opinion";
  return (
    <li className={`rounded-lg border p-3 ${opinion ? "border-emerald-200 bg-emerald-50/50" : "border-slate-200"}`}>
      <p className="text-xs text-slate-500">
        <span className="font-medium text-slate-800">{m.author_name ?? "Unknown"}</span> ·{" "}
        {m.author_role === "requester" ? "requesting clinician" : "consulting clinician"} · {formatTime(m.created_at)}
        {opinion && <span className="ml-1 chip bg-emerald-100 text-emerald-800">Professional opinion (advisory)</span>}
      </p>
      {opinion ? (
        <dl className="mt-1.5 space-y-1.5 text-sm">
          {Object.entries(OPINION_LABELS)
            .filter(([k]) => m.content[k])
            .map(([k, label]) => (
              <div key={k}>
                <dt className="text-xs font-semibold text-slate-600">{label}</dt>
                <dd className="whitespace-pre-wrap text-slate-800">{m.content[k]}</dd>
              </div>
            ))}
        </dl>
      ) : (
        <p className="mt-1 whitespace-pre-wrap text-sm text-slate-800">{m.content.text}</p>
      )}
    </li>
  );
}

function Composer({
  token,
  id,
  allowOpinion,
  onPosted,
}: {
  token: string;
  id: string;
  allowOpinion: boolean;
  onPosted: (c: ConsultDetail) => void;
}) {
  const [kind, setKind] = useState<"message" | "opinion">("message");
  const [fields, setFields] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const identifiers = findIdentifiers(...Object.values(fields));
  const ready = kind === "message" ? (fields.text ?? "").trim().length > 0 : (fields.interpretation ?? "").trim().length > 0;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!ready || identifiers.length) return;
    setBusy(true);
    setError(null);
    try {
      const body: Record<string, string> = { kind };
      if (kind === "message") body.text = fields.text ?? "";
      else for (const k of Object.keys(OPINION_LABELS)) if (fields[k]) body[k] = fields[k];
      onPosted(await peerApi.message(token, id, body));
      setFields({});
    } catch (err) {
      setError(errorText(err, "The message was not sent."));
    } finally {
      setBusy(false);
    }
  }

  const area = (k: string, placeholder: string, rows = 3) => (
    <textarea
      className={`${inputClass}`}
      rows={rows}
      placeholder={placeholder}
      aria-label={placeholder}
      value={fields[k] ?? ""}
      maxLength={4000}
      onChange={(e) => setFields({ ...fields, [k]: e.target.value })}
    />
  );

  return (
    <form onSubmit={submit} className="space-y-2 rounded-lg border border-slate-200 p-3" noValidate>
      {allowOpinion && (
        <div className="flex gap-1 rounded-lg bg-slate-100 p-1 text-sm" role="tablist">
          {(["message", "opinion"] as const).map((k) => (
            <button
              key={k}
              type="button"
              role="tab"
              aria-selected={kind === k}
              onClick={() => setKind(k)}
              className={`flex-1 rounded-md px-3 py-1 ${kind === k ? "bg-white shadow-sm" : "text-slate-600"}`}
            >
              {k === "message" ? "Message" : "Provide opinion"}
            </button>
          ))}
        </div>
      )}
      {kind === "message" ? (
        area("text", "Write a message (no identifying details)")
      ) : (
        <>
          {area("interpretation", "Your interpretation (required)", 4)}
          {area("considerations", "Relevant considerations")}
          {area("questions", "Questions for the requesting clinician", 2)}
          {area("limitations", "Limitations of this opinion", 2)}
          {area("references", "References", 2)}
          <p className="text-xs text-slate-500">
            Your opinion is advisory and will be shown as such. It does not change the treatment plan.
          </p>
        </>
      )}
      {identifiers.length > 0 && <ErrorBox message={`This looks like it contains ${identifiers.join(", ")}.`} />}
      <ErrorBox message={error} />
      <button type="submit" className={primaryButton} disabled={busy || !ready || identifiers.length > 0}>
        {busy ? "Sending…" : kind === "message" ? "Send message" : "Post opinion"}
      </button>
    </form>
  );
}
