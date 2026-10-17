"use client";

import { useCallback, useEffect, useState } from "react";
import { ApiError } from "@/lib/api";
import { peerApi, type ClinicianProfile, type Page, type PeerStatus } from "@/lib/peer";
import { Empty, ErrorBox, Pager, VerificationBadge, inputClass, primaryButton, secondaryButton } from "./ui";

/** Searchable directory of verified clinicians (the server returns verified accounts only). */
export function PeerDirectory({
  token,
  status,
  onRequest,
}: {
  token: string;
  status: PeerStatus;
  onRequest: (c: ClinicianProfile) => void;
}) {
  const [filters, setFilters] = useState({
    q: "",
    specialty: "",
    subspecialty: "",
    min_experience: "",
    interest: "",
    available: false,
  });
  const [applied, setApplied] = useState(filters);
  const [page, setPage] = useState(1);
  const [data, setData] = useState<Page<ClinicianProfile> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(
        await peerApi.clinicians(token, {
          q: applied.q.trim() || undefined,
          specialty: applied.specialty || undefined,
          subspecialty: applied.subspecialty.trim() || undefined,
          min_experience: applied.min_experience || undefined,
          interest: applied.interest.trim() || undefined,
          available: applied.available ? true : undefined,
          page,
          page_size: 12,
        }),
      );
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load the directory.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [token, applied, page]);

  useEffect(() => {
    load();
  }, [load]);

  function search(e: React.FormEvent) {
    e.preventDefault();
    setPage(1);
    setApplied(filters);
  }

  return (
    <section className="card p-5">
      <h2 className="text-base font-semibold text-slate-900">Find a Colleague</h2>
      <p className="mb-3 text-xs text-slate-500">
        Only clinicians with a verification status the server accepts are listed. Check each
        profile&rsquo;s verification label: a development override is not a licensing check.
      </p>
      <form onSubmit={search} className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        <input
          className={inputClass}
          placeholder="Search name, interests, qualifications…"
          aria-label="Search"
          value={filters.q}
          onChange={(e) => setFilters({ ...filters, q: e.target.value })}
        />
        <select
          className={inputClass}
          aria-label="Specialty"
          value={filters.specialty}
          onChange={(e) => setFilters({ ...filters, specialty: e.target.value })}
        >
          <option value="">Any specialty</option>
          {status.specialties?.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <input
          className={inputClass}
          placeholder="Subspecialty"
          aria-label="Subspecialty"
          value={filters.subspecialty}
          onChange={(e) => setFilters({ ...filters, subspecialty: e.target.value })}
        />
        <input
          className={inputClass}
          placeholder="Clinical interest"
          aria-label="Clinical interest"
          value={filters.interest}
          onChange={(e) => setFilters({ ...filters, interest: e.target.value })}
        />
        <input
          className={inputClass}
          type="number"
          min={0}
          max={70}
          placeholder="Minimum years of experience"
          aria-label="Minimum years of experience"
          value={filters.min_experience}
          onChange={(e) => setFilters({ ...filters, min_experience: e.target.value })}
        />
        <div className="flex items-center justify-between gap-2">
          <label className="flex items-center gap-2 text-sm text-slate-700">
            <input
              type="checkbox"
              checked={filters.available}
              onChange={(e) => setFilters({ ...filters, available: e.target.checked })}
            />
            Accepting requests
          </label>
          <button type="submit" className={primaryButton}>
            Search
          </button>
        </div>
      </form>

      <div className="mt-4 space-y-3">
        <ErrorBox message={error} />
        {loading && <div className="skeleton h-24 w-full" />}
        {!loading && data && data.items.length === 0 && (
          <Empty
            title="No verified clinicians match"
            body={
              data.total === 0 && !Object.values(applied).some(Boolean)
                ? "No other clinician has a verified account on this server yet."
                : "Try widening the filters."
            }
          />
        )}
        {!loading && data && data.items.length > 0 && (
          <div className="grid gap-3 md:grid-cols-2">
            {data.items.map((c) => (
              <ClinicianCard key={c.id} c={c} onRequest={() => onRequest(c)} />
            ))}
          </div>
        )}
        {data && <Pager page={data.page} pageSize={data.page_size} total={data.total} onPage={setPage} />}
      </div>
    </section>
  );
}

function ClinicianCard({ c, onRequest }: { c: ClinicianProfile; onRequest: () => void }) {
  return (
    <article className="rounded-lg border border-slate-200 p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">{c.display_name}</h3>
          <p className="text-xs text-slate-600">
            {c.specialty}
            {c.subspecialty ? ` · ${c.subspecialty}` : ""}
          </p>
        </div>
        <VerificationBadge v={c.verification} />
      </div>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
        {c.qualifications && (
          <>
            <dt className="text-slate-400">Qualifications</dt>
            <dd className="text-slate-700">{c.qualifications}</dd>
          </>
        )}
        {c.years_experience != null && (
          <>
            <dt className="text-slate-400">Experience</dt>
            <dd className="text-slate-700">{c.years_experience} years</dd>
          </>
        )}
        {c.registration_jurisdiction && (
          <>
            <dt className="text-slate-400">Registered in</dt>
            <dd className="text-slate-700">{c.registration_jurisdiction}</dd>
          </>
        )}
      </dl>
      {c.interests.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {c.interests.map((i) => (
            <span key={i} className="chip border border-slate-200 bg-white text-slate-600">
              {i}
            </span>
          ))}
        </div>
      )}
      <div className="mt-3 flex items-center justify-between gap-2">
        <span className={`text-xs ${c.accepting_consults ? "text-emerald-700" : "text-slate-400"}`}>
          {c.accepting_consults ? "Accepting requests" : "Not accepting requests"}
        </span>
        <button type="button" className={secondaryButton} disabled={!c.accepting_consults} onClick={onRequest}>
          Request consultation
        </button>
      </div>
    </article>
  );
}
