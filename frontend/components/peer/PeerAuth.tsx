"use client";

import { useState } from "react";
import { ApiError } from "@/lib/api";
import { peerApi, type PeerStatus, type SelfProfile } from "@/lib/peer";
import { ErrorBox, Field, VerificationBadge, inputClass, primaryButton, secondaryButton } from "./ui";

function message(err: unknown): string {
  return err instanceof ApiError || err instanceof Error ? err.message : "Something went wrong.";
}

/** Sign in, or create an account (which starts unverified). */
export function PeerAuth({
  status,
  onSignedIn,
}: {
  status: PeerStatus;
  onSignedIn: (token: string, profile: SelfProfile) => void;
}) {
  const [mode, setMode] = useState<"signin" | "register">("signin");
  return (
    <section className="card mx-auto max-w-xl p-6">
      <div className="mb-4 flex gap-1 rounded-lg bg-slate-100 p-1" role="tablist">
        {(["signin", "register"] as const).map((m) => (
          <button
            key={m}
            type="button"
            role="tab"
            aria-selected={mode === m}
            onClick={() => setMode(m)}
            className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium ${
              mode === m ? "bg-white text-slate-900 shadow-sm" : "text-slate-600"
            }`}
          >
            {m === "signin" ? "Sign in" : "Create clinician account"}
          </button>
        ))}
      </div>
      {mode === "signin" ? (
        <SignIn onSignedIn={onSignedIn} />
      ) : (
        <Register status={status} onRegistered={() => setMode("signin")} />
      )}
    </section>
  );
}

function SignIn({ onSignedIn }: { onSignedIn: (token: string, profile: SelfProfile) => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await peerApi.login(email, password);
      onSignedIn(res.token, res.profile);
    } catch (err) {
      setError(message(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-3" noValidate>
      <Field label="Email" required>
        <input className={inputClass} type="email" autoComplete="username" value={email} onChange={(e) => setEmail(e.target.value)} />
      </Field>
      <Field label="Password" required>
        <input className={inputClass} type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} />
      </Field>
      <ErrorBox message={error} />
      <button type="submit" className={primaryButton} disabled={busy || !email || !password}>
        {busy ? "Signing in…" : "Sign in"}
      </button>
    </form>
  );
}

function Register({ status, onRegistered }: { status: PeerStatus; onRegistered: () => void }) {
  const [form, setForm] = useState({
    display_name: "",
    email: "",
    password: "",
    specialty: "",
    subspecialty: "",
    qualifications: "",
    years_experience: "",
    registration_jurisdiction: "",
    registration_number: "",
    interests: "",
    accepting_consults: true,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm({ ...form, [k]: e.target.type === "checkbox" ? (e.target as HTMLInputElement).checked : e.target.value });

  const problems: string[] = [];
  if (form.display_name.trim().length < 3) problems.push("professional name");
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(form.email)) problems.push("a valid email");
  if (form.password.length < 10) problems.push("a password of at least 10 characters");
  if (!form.specialty) problems.push("specialty");

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (problems.length) return;
    setBusy(true);
    setError(null);
    try {
      await peerApi.register({
        display_name: form.display_name.trim(),
        email: form.email.trim(),
        password: form.password,
        specialty: form.specialty,
        subspecialty: form.subspecialty.trim() || undefined,
        qualifications: form.qualifications.trim() || undefined,
        years_experience: form.years_experience ? Number(form.years_experience) : undefined,
        registration_jurisdiction: form.registration_jurisdiction.trim() || undefined,
        registration_number: form.registration_number.trim() || undefined,
        interests: form.interests.split(",").map((s) => s.trim()).filter(Boolean),
        accepting_consults: form.accepting_consults,
      });
      setDone(true);
    } catch (err) {
      setError(message(err));
    } finally {
      setBusy(false);
    }
  }

  if (done) {
    return (
      <div className="space-y-3 text-sm text-slate-700">
        <p className="font-medium text-slate-900">Account created.</p>
        <p>
          Your account starts <strong>unverified</strong>. Peer Consult opens once your professional
          registration has been verified with a licensing authority.
        </p>
        <button type="button" className={primaryButton} onClick={onRegistered}>
          Continue to sign in
        </button>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="space-y-3" noValidate>
      <p className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
        Creating an account does not verify you. Email confirmation alone does not establish medical
        licensure, so your account stays unverified until your registration is checked with a
        licensing authority.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Professional name" required>
          <input className={inputClass} value={form.display_name} onChange={set("display_name")} maxLength={120} />
        </Field>
        <Field label="Email" required>
          <input className={inputClass} type="email" autoComplete="username" value={form.email} onChange={set("email")} />
        </Field>
        <Field label="Password" required hint="At least 10 characters.">
          <input className={inputClass} type="password" autoComplete="new-password" value={form.password} onChange={set("password")} />
        </Field>
        <Field label="Specialty" required>
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
        <Field label="Qualifications">
          <input className={inputClass} value={form.qualifications} onChange={set("qualifications")} maxLength={300} />
        </Field>
        <Field label="Years of experience">
          <input className={inputClass} type="number" min={0} max={70} value={form.years_experience} onChange={set("years_experience")} />
        </Field>
        <Field label="Clinical interests" hint="Comma-separated.">
          <input className={inputClass} value={form.interests} onChange={set("interests")} />
        </Field>
        <Field label="Registration jurisdiction">
          <input className={inputClass} value={form.registration_jurisdiction} onChange={set("registration_jurisdiction")} maxLength={100} />
        </Field>
        <Field label="Registration number">
          <input className={inputClass} value={form.registration_number} onChange={set("registration_number")} maxLength={60} />
        </Field>
      </div>
      <label className="flex items-center gap-2 text-sm text-slate-700">
        <input type="checkbox" checked={form.accepting_consults} onChange={set("accepting_consults")} />
        Accept consultation requests from colleagues
      </label>
      {problems.length > 0 && (
        <p className="text-xs text-slate-500">Still needed: {problems.join(", ")}.</p>
      )}
      <ErrorBox message={error} />
      <button type="submit" className={primaryButton} disabled={busy || problems.length > 0}>
        {busy ? "Creating…" : "Create account"}
      </button>
    </form>
  );
}

/** Shown to signed-in clinicians who cannot use Peer Consult yet. */
export function VerificationPanel({
  token,
  status,
  profile,
  onProfile,
}: {
  token: string;
  status: PeerStatus;
  profile: SelfProfile;
  onProfile: (p: SelfProfile) => void;
}) {
  const [jurisdiction, setJurisdiction] = useState(profile.registration_jurisdiction ?? "");
  const [number, setNumber] = useState(profile.registration_number ?? "");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await peerApi.requestVerification(token, jurisdiction.trim(), number.trim());
      onProfile(res.profile);
      setResult(res.detail);
    } catch (err) {
      setError(message(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card space-y-4 p-6">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-base font-semibold text-slate-900">Verification required</h2>
        <VerificationBadge v={profile.verification} />
      </div>
      <p className="text-sm text-slate-700">
        Peer Consult is available only to clinicians whose professional registration has been
        verified with the licensing authority in their jurisdiction. This is checked by the server,
        so the directory, requests and messages stay closed until then.
      </p>
      {profile.verification.status === "dev_override" && (
        <p className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          This account has a development override, but this server does not accept overrides.
        </p>
      )}
      {status.verification && !status.verification.available && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
          <p className="font-medium">Licence verification integration: incomplete</p>
          <p className="mt-0.5">{status.verification.message}</p>
        </div>
      )}
      <form onSubmit={submit} className="grid gap-3 sm:grid-cols-[1fr_1fr_auto] sm:items-end" noValidate>
        <Field label="Registration jurisdiction" required>
          <input className={inputClass} value={jurisdiction} onChange={(e) => setJurisdiction(e.target.value)} />
        </Field>
        <Field label="Registration number" required>
          <input className={inputClass} value={number} onChange={(e) => setNumber(e.target.value)} />
        </Field>
        <button type="submit" className={secondaryButton} disabled={busy || jurisdiction.trim().length < 2 || number.trim().length < 2}>
          {busy ? "Submitting…" : "Submit for verification"}
        </button>
      </form>
      {result && <p className="rounded-lg bg-slate-50 p-3 text-sm text-slate-700">Result: {result}</p>}
      <ErrorBox message={error} />
      {status.verification?.dev_override_enabled && (
        <p className="text-xs text-slate-500">
          Development server: an administrator can set a clearly labelled development override with{" "}
          <span className="font-mono">python scripts/peer_consult_admin.py dev-verify &lt;email&gt;</span>. It is not a
          licensing check.
        </p>
      )}
    </section>
  );
}
