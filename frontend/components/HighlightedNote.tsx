"use client";

import { useMemo, useState } from "react";
import type { ExplanationPayload, WordScore } from "@/lib/types";

interface Props {
  text: string;
  explanation: ExplanationPayload;
  /** Words scoring below this fraction of the peak are left unhighlighted. */
  threshold?: number;
}

/**
 * Renders the note with each word tinted by its importance.
 *
 * Two colour rules, both of which exist to avoid lying to the reader:
 *
 * 1. **Signed methods get a diverging scale** (red = raises risk, blue = lowers
 *    it). **Unsigned methods get a single-hue scale**, because attention has no
 *    direction — showing a low-attention word in "protective blue" would invent
 *    a claim the method cannot make.
 * 2. **Intensity is scaled to the note's own peak**, not to a fixed absolute
 *    range. Attributions differ in magnitude by orders of magnitude between
 *    methods and notes, so a fixed scale renders most notes uniformly blank.
 */
export function HighlightedNote({ text, explanation, threshold = 0.08 }: Props) {
  const [hovered, setHovered] = useState<number | null>(null);

  const segments = useMemo(() => buildSegments(text, explanation.words), [text, explanation.words]);
  const peak = useMemo(
    () => Math.max(...explanation.words.map((w) => Math.abs(w.score)), 1e-9),
    [explanation.words],
  );

  return (
    <div className="relative">
      <div className="note-surface rounded-lg bg-white p-4 font-mono text-[13px] text-slate-800">
        {segments.map((segment, index) =>
          segment.word ? (
            <span
              key={index}
              className="note-token"
              data-interactive="true"
              style={{ backgroundColor: tint(segment.word.score / peak, explanation.signed, threshold) }}
              onMouseEnter={() => setHovered(index)}
              onMouseLeave={() => setHovered(null)}
              title={`${segment.text}: ${formatScore(segment.word.score, explanation.signed)}`}
            >
              {segment.text}
            </span>
          ) : (
            <span key={index}>{segment.text}</span>
          ),
        )}
      </div>

      {hovered !== null && segments[hovered]?.word && (
        <div className="pointer-events-none absolute bottom-2 right-2 rounded-md bg-slate-900/90 px-3 py-1.5 text-xs text-white shadow-lg">
          <span className="font-mono">{segments[hovered].text}</span>
          <span className="ml-2 text-slate-300">
            {formatScore(segments[hovered].word!.score, explanation.signed)}
          </span>
        </div>
      )}
    </div>
  );
}

export function HighlightLegend({ signed }: { signed: boolean }) {
  return (
    <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500">
      {signed ? (
        <>
          <LegendSwatch label="Lowers risk" colors={["#bfdbfe", "#3b82f6"]} reverse />
          <LegendSwatch label="Raises risk" colors={["#fecaca", "#dc2626"]} />
          <span>Colour intensity is relative to the strongest word in this note.</span>
        </>
      ) : (
        <>
          <LegendSwatch label="Used by the model" colors={["#ddd6fe", "#7c3aed"]} />
          <span>
            This method is unsigned: intensity means &ldquo;influenced the prediction&rdquo;, not
            &ldquo;raised the risk&rdquo;.
          </span>
        </>
      )}
    </div>
  );
}

function LegendSwatch({
  label,
  colors,
  reverse = false,
}: {
  label: string;
  colors: [string, string];
  reverse?: boolean;
}) {
  const gradient = reverse
    ? `linear-gradient(90deg, ${colors[1]}, ${colors[0]})`
    : `linear-gradient(90deg, ${colors[0]}, ${colors[1]})`;
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-3 w-12 rounded" style={{ background: gradient }} />
      {label}
    </span>
  );
}

interface Segment {
  text: string;
  word?: WordScore;
}

/**
 * Splits the note into word / non-word segments using the character offsets the
 * backend returned. Offsets are used rather than re-tokenising in the browser so
 * the highlight lands on exactly the span the model scored — re-splitting on
 * whitespace here would drift from the server's word boundaries.
 */
function buildSegments(text: string, words: WordScore[]): Segment[] {
  const ordered = [...words].filter((w) => w.start >= 0 && w.end > w.start).sort((a, b) => a.start - b.start);
  const segments: Segment[] = [];
  let cursor = 0;

  for (const word of ordered) {
    if (word.start < cursor) continue; // defensive: ignore overlaps
    if (word.start > cursor) segments.push({ text: text.slice(cursor, word.start) });
    segments.push({ text: text.slice(word.start, word.end), word });
    cursor = word.end;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor) });
  return segments;
}

function tint(relative: number, signed: boolean, threshold: number): string {
  const magnitude = Math.abs(relative);
  if (magnitude < threshold) return "transparent";
  // gamma < 1 lifts mid-range words out of the background; without it a single
  // dominant token makes every other highlight invisible
  const alpha = Math.min(0.85, Math.pow(magnitude, 0.65) * 0.9);

  if (!signed) return `rgba(124, 58, 237, ${alpha})`;
  return relative > 0 ? `rgba(220, 38, 38, ${alpha})` : `rgba(37, 99, 235, ${alpha})`;
}

function formatScore(score: number, signed: boolean): string {
  const value = score.toFixed(3);
  if (!signed) return `importance ${Math.abs(score).toFixed(3)}`;
  return score > 0 ? `+${value} (raises risk)` : `${value} (lowers risk)`;
}
