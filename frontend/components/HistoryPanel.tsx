"use client";

import { useEffect, useState } from "react";
import { clearHistory, getHistory } from "@/lib/api";
import type { HistoryEntry } from "@/lib/types";

/**
 * Session log of recent predictions.
 *
 * The backend keeps this in memory only and never writes note text to disk, so
 * the log dies with the process. That is deliberate for a tool that people will
 * paste clinical text into.
 */
export function HistoryPanel({ version }: { version: number }) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [open, setOpen] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getHistory()
      .then((rows) => !cancelled && setEntries(rows))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [version]);

  return (
    <div className="card">
      <div className="card-header">
        <button
          type="button"
          className="flex items-center gap-1.5 text-left"
          onClick={() => setOpen((v) => !v)}
        >
          <span className={`text-slate-400 transition-transform ${open ? "rotate-90" : ""}`}>›</span>
          <h2 className="card-title">Session log ({entries.length})</h2>
        </button>
        {entries.length > 0 && (
          <button
            type="button"
            className="btn-ghost !px-2 !py-1 text-xs"
            onClick={async () => {
              await clearHistory().catch(() => undefined);
              setEntries([]);
            }}
          >
            Clear
          </button>
        )}
      </div>

      {open && (
        <div className="max-h-64 overflow-y-auto">
          {entries.length === 0 ? (
            <p className="px-4 py-4 text-xs text-slate-400">
              Predictions made in this session appear here. Nothing is written to disk.
            </p>
          ) : (
            <ul className="divide-y divide-slate-100">
              {entries.map((entry) => (
                <li key={`${entry.note_id}-${entry.timestamp}`} className="px-4 py-2.5">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="font-mono text-xs font-semibold text-slate-700">
                      {(entry.probability * 100).toFixed(1)}%
                    </span>
                    <span className="text-[10px] text-slate-400">
                      {new Date(entry.timestamp * 1000).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="mt-0.5 truncate text-[11px] text-slate-500">{entry.preview}</p>
                  <p className="mt-0.5 font-mono text-[10px] text-slate-400">
                    {entry.model} · {entry.method}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
