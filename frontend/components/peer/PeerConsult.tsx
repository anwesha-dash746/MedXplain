"use client";

import { useCallback, useEffect, useState } from "react";
import { ApiError } from "@/lib/api";
import {
  loadToken,
  peerApi,
  saveToken,
  type Box,
  type ClinicianProfile,
  type PeerStatus,
  type SelfProfile,
} from "@/lib/peer";
import { PeerAuth, VerificationPanel } from "./PeerAuth";
import { ConsultationDetail, ConsultationList } from "./PeerConsultations";
import { PeerDirectory } from "./PeerDirectory";
import { PeerRequestForm } from "./PeerRequestForm";
import { EmergencyNotice, ErrorBox, VerificationBadge, secondaryButton } from "./ui";

type View = "home" | "new" | "directory" | Box | { detail: string };

/**
 * Peer Consult tab. Everything shown comes from /api/peer; counts are the sizes
 * of the server's own lists, and every permission is enforced server-side.
 *
 * `openNewRequest` changes whenever the Results tab's "Request Peer
 * Consultation" button is pressed, which opens an empty request form. Nothing
 * from the analysed note is passed in.
 */
export function PeerConsult({ openNewRequest }: { openNewRequest: number }) {
  const [status, setStatus] = useState<PeerStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [profile, setProfile] = useState<SelfProfile | null>(null);
  const [checkingSession, setCheckingSession] = useState(true);
  const [view, setView] = useState<View>("home");
  const [preferred, setPreferred] = useState<ClinicianProfile | null>(null);
  const [counts, setCounts] = useState<Record<Box, number> | null>(null);
  const [countsError, setCountsError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [formKey, setFormKey] = useState(0);
  const [lastList, setLastList] = useState<Box>("sent");

  useEffect(() => {
    peerApi
      .status()
      .then(setStatus)
      .catch((err) => setStatusError(err instanceof ApiError ? err.message : "Peer Consult is unavailable."));
    const saved = loadToken();
    if (!saved) {
      setCheckingSession(false);
      return;
    }
    peerApi
      .me(saved)
      .then((p) => {
        setToken(saved);
        setProfile(p);
      })
      .catch(() => saveToken(null))
      .finally(() => setCheckingSession(false));
  }, []);

  useEffect(() => {
    if (openNewRequest > 0) {
      setPreferred(null);
      setFormKey((k) => k + 1);
      setView("new");
    }
  }, [openNewRequest]);

  const canUse = Boolean(profile?.can_use_peer_consult);

  const refreshCounts = useCallback(async () => {
    if (!token || !canUse) return;
    try {
      setCounts((await peerApi.dashboard(token)).counts);
      setCountsError(null);
    } catch (err) {
      setCountsError(err instanceof ApiError ? err.message : "Could not load counts.");
    }
  }, [token, canUse]);

  useEffect(() => {
    refreshCounts();
  }, [refreshCounts, view]);

  function signedIn(t: string, p: SelfProfile) {
    saveToken(t);
    setToken(t);
    setProfile(p);
  }

  async function signOut() {
    if (token) await peerApi.logout(token).catch(() => undefined);
    saveToken(null);
    setToken(null);
    setProfile(null);
    setCounts(null);
    setView("home");
  }

  const header = (
    <div className="card p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold text-slate-900">Peer Consult</h2>
          <p className="text-sm text-slate-600">
            Connect with verified clinicians to discuss complex cases and gain another professional perspective.
          </p>
        </div>
        {profile && (
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="text-slate-700">{profile.display_name}</span>
            <VerificationBadge v={profile.verification} />
            <button type="button" className={secondaryButton} onClick={signOut}>
              Sign out
            </button>
          </div>
        )}
      </div>
      <div className="mt-3">
        <EmergencyNotice />
      </div>
    </div>
  );

  if (statusError || (status && !status.database.available)) {
    return (
      <div className="mx-auto max-w-5xl space-y-4">
        {header}
        <section className="card p-6">
          <h3 className="text-base font-semibold text-slate-900">Peer Consult is not available</h3>
          <p className="mt-1 text-sm text-slate-600">
            The consultation database could not be reached, so no requests, colleagues or messages can be
            shown. Nothing is displayed in their place.
          </p>
          <p className="mt-2 font-mono text-xs text-slate-500">{statusError ?? status?.database.error}</p>
        </section>
      </div>
    );
  }
  if (!status || checkingSession) {
    return (
      <div className="mx-auto max-w-5xl space-y-4">
        {header}
        <div className="card p-6"><div className="skeleton h-32 w-full" /></div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      {header}
      <IntegrationNotes status={status} />

      {!token || !profile ? (
        <PeerAuth status={status} onSignedIn={signedIn} />
      ) : !canUse ? (
        <VerificationPanel token={token} status={status} profile={profile} onProfile={setProfile} />
      ) : (
        <>
          <Dashboard
            counts={counts}
            error={countsError}
            active={view}
            onSelect={(v) => {
              setNotice(null);
              if (v === "new") {
                setPreferred(null);
                setFormKey((k) => k + 1);
              }
              setView(v);
            }}
          />
          {notice && (
            <div role="status" className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-2.5 text-sm text-emerald-900">
              {notice}
            </div>
          )}
          {view === "home" && (
            <section className="card p-5 text-sm text-slate-600">
              Choose an option above. Requests and messages are visible only to the clinicians involved.
            </section>
          )}
          {view === "new" && (
            <PeerRequestForm
              key={formKey}
              token={token}
              status={status}
              preferred={preferred}
              onCreated={(c) => {
                setNotice(`Request sent. ${c.assigned ? `${c.assigned.name} has been asked to respond.` : "It is now open to verified clinicians in that specialty."}`);
                setView({ detail: c.id });
              }}
            />
          )}
          {view === "directory" && (
            <PeerDirectory
              token={token}
              status={status}
              onRequest={(c) => {
                setPreferred(c);
                setFormKey((k) => k + 1);
                setView("new");
              }}
            />
          )}
          {typeof view === "string" && ["sent", "received", "active", "completed"].includes(view) && (
            <ConsultationList
              token={token}
              box={view as Box}
              onOpen={(id) => {
                setLastList(view as Box);
                setView({ detail: id });
              }}
            />
          )}
          {typeof view === "object" && (
            <ConsultationDetail
              key={view.detail}
              token={token}
              id={view.detail}
              onBack={() => setView(lastList)}
              onChanged={refreshCounts}
            />
          )}
        </>
      )}
    </div>
  );
}

function IntegrationNotes({ status }: { status: PeerStatus }) {
  const v = status.verification;
  return (
    <details className="card px-5 py-3 text-sm">
      <summary className="cursor-pointer select-none font-medium text-slate-700">
        System status
        {v && !v.available && <span className="ml-2 chip bg-amber-100 text-amber-900">Licence verification: incomplete</span>}
      </summary>
      <ul className="mt-2 space-y-1 text-xs text-slate-600">
        <li>Consultation storage: available (schema v{status.database.schema_version}).</li>
        {v && (
          <li>
            Licence verification: {v.available ? "connected" : "not connected"}. {v.message}
            {v.dev_override_enabled && " Development overrides are accepted on this server and are labelled as such."}
          </li>
        )}
        {status.encryption && <li>Encryption: {status.encryption.note}</li>}
        {status.attachments && <li>Attachments: {status.attachments.note}</li>}
        <li>This is a research prototype. No regulatory compliance is claimed.</li>
      </ul>
    </details>
  );
}

function Dashboard({
  counts,
  error,
  active,
  onSelect,
}: {
  counts: Record<Box, number> | null;
  error: string | null;
  active: View;
  onSelect: (v: View) => void;
}) {
  const cards: { view: View; title: string; body: string; count?: number | null }[] = [
    { view: "new", title: "New Consultation Request", body: "Ask a colleague a specific clinical question." },
    { view: "directory", title: "Find a Colleague", body: "Search verified clinicians by specialty and interests." },
    { view: "sent", title: "My Requests", body: "Requests you have sent.", count: counts?.sent },
    { view: "received", title: "Requests Received", body: "Waiting for your response.", count: counts?.received },
    { view: "active", title: "In Discussion", body: "Accepted and ongoing.", count: counts?.active },
    { view: "completed", title: "Completed", body: "Closed, declined or cancelled.", count: counts?.completed },
  ];
  const isActive = (v: View) => (typeof active === "string" ? active === v : false);
  return (
    <div className="space-y-2">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {cards.map((c) => (
          <button
            key={String(c.view)}
            type="button"
            onClick={() => onSelect(c.view)}
            aria-pressed={isActive(c.view)}
            className={`card flex items-start justify-between gap-3 p-4 text-left transition-colors hover:border-clinical-300 ${
              isActive(c.view) ? "border-clinical-500 ring-2 ring-clinical-100" : ""
            }`}
          >
            <span>
              <span className="block text-sm font-semibold text-slate-900">{c.title}</span>
              <span className="block text-xs text-slate-500">{c.body}</span>
            </span>
            {c.count !== undefined && (
              <span className="text-2xl font-semibold text-slate-900">{c.count ?? "–"}</span>
            )}
          </button>
        ))}
      </div>
      <ErrorBox message={error} />
    </div>
  );
}
