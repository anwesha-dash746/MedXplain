"use client";

import type { ConsultStatus, Verification } from "@/lib/peer";

export const EMERGENCY_NOTICE =
  "Peer Consult is not an emergency service. For urgent or emergency patient care, use established clinical escalation channels.";

const STATUS_STYLE: Record<ConsultStatus, string> = {
  requested: "bg-sky-100 text-sky-800",
  accepted: "bg-indigo-100 text-indigo-800",
  in_discussion: "bg-violet-100 text-violet-800",
  response_received: "bg-emerald-100 text-emerald-800",
  closed: "bg-slate-200 text-slate-700",
  declined: "bg-rose-100 text-rose-800",
  cancelled: "bg-slate-100 text-slate-500",
};

export function StatusChip({ status, label }: { status: ConsultStatus; label: string }) {
  return <span className={`chip ${STATUS_STYLE[status] ?? "bg-slate-100 text-slate-700"}`}>{label}</span>;
}

export function VerificationBadge({ v }: { v: Verification }) {
  const style = v.authoritative
    ? "border-emerald-200 bg-emerald-50 text-emerald-800"
    : v.status === "dev_override"
      ? "border-amber-300 bg-amber-50 text-amber-900"
      : "border-slate-300 bg-slate-50 text-slate-600";
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${style}`} title={v.note ?? ""}>
      {v.authoritative ? "✓ " : ""}
      {v.label}
    </span>
  );
}

export function EmergencyNotice() {
  return (
    <div role="note" className="rounded-lg border border-red-200 bg-red-50 px-4 py-2.5 text-sm text-red-800">
      <strong>Peer Consult is not an emergency service.</strong> For urgent or emergency patient care, use
      established clinical escalation channels.
    </div>
  );
}

export function ErrorBox({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
      {message}
    </div>
  );
}

export function Field({
  label,
  hint,
  required,
  error,
  children,
}: {
  label: string;
  hint?: string;
  required?: boolean;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-slate-700">
        {label}
        {required ? <span className="text-red-600"> *</span> : <span className="font-normal text-slate-400"> (optional)</span>}
      </span>
      {hint && <span className="mt-0.5 block text-xs text-slate-500">{hint}</span>}
      <div className="mt-1">{children}</div>
      {error && <span className="mt-1 block text-xs text-red-700">{error}</span>}
    </label>
  );
}

export const inputClass =
  "w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus:border-clinical-500 focus:outline-none focus:ring-2 focus:ring-clinical-100 disabled:bg-slate-50";

export const primaryButton =
  "rounded-lg bg-clinical-600 px-4 py-2 text-sm font-medium text-white hover:bg-clinical-700 disabled:cursor-not-allowed disabled:opacity-50";

export const secondaryButton =
  "rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50";

export function Empty({ title, body }: { title: string; body?: string }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 p-6 text-center">
      <p className="text-sm font-medium text-slate-700">{title}</p>
      {body && <p className="mt-1 text-xs text-slate-500">{body}</p>}
    </div>
  );
}

export function formatTime(ts: number | null | undefined): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function Pager({
  page,
  pageSize,
  total,
  onPage,
}: {
  page: number;
  pageSize: number;
  total: number;
  onPage: (p: number) => void;
}) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  if (pages <= 1) return null;
  return (
    <div className="flex items-center justify-between pt-2 text-xs text-slate-500">
      <span>
        Page {page} of {pages} · {total} total
      </span>
      <div className="flex gap-2">
        <button type="button" className={secondaryButton} disabled={page <= 1} onClick={() => onPage(page - 1)}>
          Previous
        </button>
        <button type="button" className={secondaryButton} disabled={page >= pages} onClick={() => onPage(page + 1)}>
          Next
        </button>
      </div>
    </div>
  );
}
