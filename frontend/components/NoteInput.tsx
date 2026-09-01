"use client";

import { useRef, useState } from "react";
import type { ExampleNote } from "@/lib/types";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading: boolean;
  examples: ExampleNote[];
  maxChars: number;
}

export function NoteInput({ value, onChange, onSubmit, loading, examples, maxChars }: Props) {
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const chars = value.length;
  const overLimit = chars > maxChars;
  const tooShort = value.trim().length > 0 && value.trim().length < 20;

  async function readFile(file: File) {
    setFileError(null);
    if (file.size > 2 * 1024 * 1024) {
      setFileError("File is larger than 2 MB.");
      return;
    }
    if (!/\.(txt|md|csv|json|text)$/i.test(file.name) && !file.type.startsWith("text/")) {
      setFileError("Only plain-text files can be read in the browser.");
      return;
    }
    onChange(await file.text());
  }

  return (
    <div className="card flex h-full flex-col">
      <div className="card-header">
        <h2 className="card-title">Clinical note</h2>
        <div className="flex items-center gap-2">
          <button
            className="btn-ghost !px-2 !py-1 text-xs"
            onClick={() => fileInput.current?.click()}
            type="button"
          >
            Upload .txt
          </button>
          <button
            className="btn-ghost !px-2 !py-1 text-xs"
            onClick={() => onChange("")}
            disabled={!value}
            type="button"
          >
            Clear
          </button>
        </div>
      </div>

      <div
        className={`relative flex-1 ${dragging ? "ring-2 ring-clinical-400 ring-inset" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files?.[0];
          if (file) void readFile(file);
        }}
      >
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !loading) onSubmit();
          }}
          placeholder="Paste a discharge summary or progress note here, drop a .txt file, or load one of the case studies below."
          spellCheck={false}
          className="h-full min-h-[280px] w-full resize-none border-0 bg-transparent p-4 font-mono text-[13px]
                     leading-relaxed text-slate-800 placeholder:text-slate-400 focus:outline-none"
        />
        {dragging && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center bg-clinical-50/85 text-sm font-medium text-clinical-700">
            Drop the file to load it
          </div>
        )}
      </div>

      <input
        ref={fileInput}
        type="file"
        accept=".txt,.md,.csv,.json,text/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void readFile(file);
          e.target.value = "";
        }}
      />

      {examples.length > 0 && (
        <div className="border-t border-slate-200 px-4 py-3">
          <p className="field-label">Case studies (held-out test notes)</p>
          <div className="flex flex-wrap gap-1.5">
            {examples.map((example) => (
              <button
                key={example.note_id}
                type="button"
                onClick={() => onChange(example.text)}
                title={`True outcome: ${example.true_label === 1 ? "positive" : "negative"}`}
                className="chip border border-slate-300 bg-white text-slate-600 hover:border-clinical-400 hover:text-clinical-700"
              >
                {example.title}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="flex items-center justify-between gap-3 border-t border-slate-200 px-4 py-3">
        <div className="text-xs text-slate-500">
          <span className={overLimit ? "font-semibold text-red-600" : ""}>
            {chars.toLocaleString()}
          </span>
          {" / "}
          {maxChars.toLocaleString()} characters
          {fileError && <span className="ml-2 text-red-600">{fileError}</span>}
          {tooShort && <span className="ml-2 text-amber-600">Too short to score.</span>}
        </div>
        <button
          className="btn-primary"
          onClick={onSubmit}
          disabled={loading || !value.trim() || overLimit || tooShort}
          type="button"
        >
          {loading ? (
            <>
              <Spinner />
              Analysing…
            </>
          ) : (
            <>Analyse note</>
          )}
        </button>
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  );
}
