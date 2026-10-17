"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError } from "@/lib/api";
import {
  findIdentifiers,
  newRequestId,
  peerApi,
  type ClinicianProfile,
  type ConsultDetail,
  type PeerStatus,
} from "@/lib/peer";
import { EmergencyNotice, ErrorBox, Field, inputClass, primaryButton, secondaryButton } from "./ui";

interface FormState {
  title: string;
  specialty: string;
  subspecialty: string;
  summary: string;
  findings: string;
  diagnosis: string;
  question: string;
  consult_type: string;
  priority: string;
  preferred_clinician_id: string;
  matching_preferences: string;
  deidentified_confirmed: boolean;
  authorized_confirmed: boolean;
}

const EMPTY: FormState = {
  title: "",
  specialty: "",
  subspecialty: "",
  summary: "",
  findings: "",
  diagnosis: "",
  question: "",
  consult_type: "",
  priority: "routine",
  preferred_clinician_id: "",
  matching_preferences: "",
  deidentified_confirmed: false,
  authorized_confirmed: false,
};

/** Limits mirror backend/app/schemas_peer.py; the server re-validates everything. */
function validate(f: FormState): Partial<Record<keyof FormState, string>> {
  const e: Partial<Record<keyof FormState, string>> = {};
  if (f.title.trim().length < 5) e.title = "At least 5 characters.";
  if (!f.specialty) e.specialty = "Choose a specialty.";
  if (f.summary.trim().length < 30) e.summary = "At least 30 characters.";
  if (f.findings.trim().length < 5) e.findings = "At least 5 characters.";
  if (f.question.trim().length < 10) e.question = "At least 10 characters.";
  if (!f.consult_type) e.consult_type = "Choose a consultation type.";
  if (!f.deidentified_confirmed) e.deidentified_confirmed = "Required before sharing.";
  if (!f.authorized_confirmed) e.authorized_confirmed = "Required before sharing.";
  return e;
}

export function PeerRequestForm({
  token,
  status,
  preferred,
  onCreated,
}: {
  token: string;
  status: PeerStatus;
  preferred: ClinicianProfile | null;
  onCreated: (c: ConsultDetail) => void;
}) {
  const [form, setForm] = useState<FormState>({
    ...EMPTY,
    specialty: preferred?.specialty ?? "",
    preferred_clinician_id: preferred?.id ?? "",
  });
  const [touched, setTouched] = useState(false);
  const [reviewing, setReviewing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [colleagues, setColleagues] = useState<ClinicianProfile[]>(preferred ? [preferred] : []);
  // one id per form: a retried or double-clicked submit returns the original request
  const requestId = useRef(newRequestId());

  useEffect(() => {
    if (!form.specialty) return;
    let cancelled = false;
    peerApi
      .clinicians(token, { specialty: form.specialty, available: true, page_size: 50 })
      .then((page) => {
        if (cancelled) return;
        const list = page.items;
        if (preferred && preferred.specialty === form.specialty && !list.some((c) => c.id === preferred.id)) list.unshift(preferred);
        setColleagues(list);
      })
      .catch(() => !cancelled && setColleagues([]));
    return () => {
      cancelled = true;
    };
  }, [form.specialty, token, preferred]);

  const errors = validate(form);
  const identifiers = findIdentifiers(
    form.title, form.summary, form.findings, form.diagnosis, form.question, form.matching_preferences,
  );
  const set = (k: keyof FormState) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const value = e.target.type === "checkbox" ? (e.target as HTMLInputElement).checked : e.target.value;
    setForm((f) => ({ ...f, [k]: value, ...(k === "specialty" ? { preferred_clinician_id: "" } : {}) }));
    setReviewing(false);
  };
  const show = (k: keyof FormState) => (touched ? errors[k] : undefined);
  const chosen = colleagues.find((c) => c.id === form.preferred_clinician_id);

  function review(e: React.FormEvent) {
    e.preventDefault();
    setTouched(true);
    if (Object.keys(errors).length === 0 && identifiers.length === 0) setReviewing(true);
  }

  async function send() {
    setBusy(true);
    setError(null);
    try {
      const res = await peerApi.createConsultation(token, {
        client_request_id: requestId.current,
        title: form.title.trim(),
        specialty: form.specialty,
        subspecialty: form.subspecialty.trim() || undefined,
        summary: form.summary.trim(),
        findings: form.findings.trim(),
        diagnosis: form.diagnosis.trim() || undefined,
        question: form.question.trim(),
        consult_type: form.consult_type,
        priority: form.priority,
        preferred_clinician_id: form.preferred_clinician_id || undefined,
        matching_preferences: form.matching_preferences.trim() || undefined,
        deidentified_confirmed: form.deidentified_confirmed,
        authorized_confirmed: form.authorized_confirmed,
      });
      // success is shown only now, after the server has stored the request
      requestId.current = newRequestId();
      onCreated(res.consultation);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The request could not be sent.");
      setReviewing(false);
    } finally {
      setBusy(false);
    }
  }

  const typeLabel = status.consult_types?.find((t) => t.key === form.consult_type)?.label ?? "";

  return (
    <section className="card space-y-4 p-5">
      <div>
        <h2 className="text-base font-semibold text-slate-900">New Consultation Request</h2>
        <p className="text-xs text-slate-500">
          Nothing from the analysed note is copied here. Write a de-identified summary yourself.
        </p>
      </div>
      <EmergencyNotice />

      <form onSubmit={review} className="space-y-4" noValidate>
        <div className="grid gap-3 md:grid-cols-2">
          <Field label="Consultation title" required error={show("title")}>
            <input className={inputClass} value={form.title} onChange={set("title")} maxLength={150} />
          </Field>
          <Field label="Specialty" required error={show("specialty")}>
            <select className={inputClass} value={form.specialty} onChange={set("specialty")}>
              <option value="">Choose…</option>
              {status.specialties?.map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
          </Field>
          <Field label="Subspecialty">
            <input className={inputClass} value={form.subspecialty} onChange={set("subspecialty")} maxLength={100} />
          </Field>
          <Field label="Consultation type" required error={show("consult_type")}>
            <select className={inputClass} value={form.consult_type} onChange={set("consult_type")}>
              <option value="">Choose…</option>
              {status.consult_types?.map((t) => (
                <option key={t.key} value={t.key}>
                  {t.label}
                </option>
              ))}
            </select>
          </Field>
        </div>

        <Field
          label="De-identified case summary"
          required
          hint="Age band, relevant history and course. No names, dates, record numbers or locations."
          error={show("summary")}
        >
          <textarea className={`${inputClass} min-h-28`} value={form.summary} onChange={set("summary")} maxLength={4000} />
        </Field>
        <Field label="Relevant symptoms and findings" required error={show("findings")}>
          <textarea className={`${inputClass} min-h-20`} value={form.findings} onChange={set("findings")} maxLength={4000} />
        </Field>
        <Field label="Documented diagnosis or differential" hint="If known.">
          <textarea className={`${inputClass} min-h-16`} value={form.diagnosis} onChange={set("diagnosis")} maxLength={2000} />
        </Field>
        <Field label="Specific clinical question" required error={show("question")}>
          <textarea className={`${inputClass} min-h-16`} value={form.question} onChange={set("question")} maxLength={2000} />
        </Field>

        <div className="grid gap-3 md:grid-cols-2">
          <fieldset>
            <legend className="text-sm font-medium text-slate-700">
              Priority <span className="text-red-600">*</span>
            </legend>
            <div className="mt-1 flex flex-wrap gap-4">
              {status.priorities?.map((p) => (
                <label key={p.key} className="flex items-center gap-2 text-sm text-slate-700">
                  <input type="radio" name="priority" value={p.key} checked={form.priority === p.key} onChange={set("priority")} />
                  {p.label}
                </label>
              ))}
            </div>
            {form.priority === "time_sensitive" && (
              <p className="mt-1 text-xs text-amber-800">
                Time-sensitive does not mean urgent. Colleagues may not reply quickly.
              </p>
            )}
          </fieldset>
          <Field label="Preferred clinician" hint={form.specialty ? undefined : "Choose a specialty first."}>
            <select className={inputClass} value={form.preferred_clinician_id} onChange={set("preferred_clinician_id")} disabled={!form.specialty}>
              <option value="">Any verified clinician in this specialty (open request)</option>
              {colleagues.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.display_name}
                  {c.subspecialty ? ` (${c.subspecialty})` : ""}
                </option>
              ))}
            </select>
          </Field>
        </div>
        {!form.preferred_clinician_id && form.specialty && (
          <p className="text-xs text-slate-500">
            An open request is visible to every verified {form.specialty} clinician who accepts requests,
            until one of them accepts it.
          </p>
        )}
        <Field label="Matching preferences" hint="E.g. experience with a condition.">
          <input className={inputClass} value={form.matching_preferences} onChange={set("matching_preferences")} maxLength={500} />
        </Field>

        <p className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
          <strong>Attachments:</strong> {status.attachments?.note ?? "Not supported."}
        </p>

        <div className="space-y-2 rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-950">
          <p className="font-medium">Before you share</p>
          <p>
            Share only the minimum information a colleague needs to answer your question. Remove names,
            dates, record numbers, addresses and any other identifying detail. Make sure you are
            authorised to share this case under your organisation&rsquo;s policies.
          </p>
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" checked={form.deidentified_confirmed} onChange={set("deidentified_confirmed")} />
            <span>I have removed identifying information and included only what is necessary.</span>
          </label>
          {show("deidentified_confirmed") && <p className="text-xs text-red-700">{errors.deidentified_confirmed}</p>}
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" checked={form.authorized_confirmed} onChange={set("authorized_confirmed")} />
            <span>I am authorised to share these case details for a professional consultation.</span>
          </label>
          {show("authorized_confirmed") && <p className="text-xs text-red-700">{errors.authorized_confirmed}</p>}
        </div>

        {identifiers.length > 0 && (
          <ErrorBox message={`This looks like it contains ${identifiers.join(", ")}. Remove it before sharing.`} />
        )}
        {touched && Object.keys(errors).length > 0 && (
          <p className="text-sm text-red-700">Please complete the highlighted fields.</p>
        )}

        {!reviewing && (
          <button type="submit" className={primaryButton}>
            Review request
          </button>
        )}
      </form>

      {reviewing && (
        <div className="space-y-3 rounded-lg border-2 border-clinical-300 p-4">
          <p className="text-sm font-semibold text-slate-900">Review exactly what will be shared</p>
          <dl className="grid gap-x-4 gap-y-2 text-sm sm:grid-cols-[max-content_1fr]">
            <Review label="Title" value={form.title} />
            <Review label="Specialty" value={[form.specialty, form.subspecialty].filter(Boolean).join(" · ")} />
            <Review label="Type" value={typeLabel} />
            <Review label="Priority" value={form.priority === "routine" ? "Routine" : "Time-sensitive"} />
            <Review label="Sent to" value={chosen ? chosen.display_name : `All verified ${form.specialty} clinicians accepting requests`} />
            <Review label="Summary" value={form.summary} />
            <Review label="Findings" value={form.findings} />
            <Review label="Diagnosis" value={form.diagnosis || "—"} />
            <Review label="Question" value={form.question} />
            <Review label="Preferences" value={form.matching_preferences || "—"} />
          </dl>
          <ErrorBox message={error} />
          <div className="flex flex-wrap gap-2">
            <button type="button" className={primaryButton} onClick={send} disabled={busy}>
              {busy ? "Sending…" : "Confirm and send"}
            </button>
            <button type="button" className={secondaryButton} onClick={() => setReviewing(false)} disabled={busy}>
              Edit
            </button>
          </div>
        </div>
      )}
      {!reviewing && <ErrorBox message={error} />}
    </section>
  );
}

function Review({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-slate-500">{label}</dt>
      <dd className="whitespace-pre-wrap text-slate-800">{value}</dd>
    </>
  );
}
