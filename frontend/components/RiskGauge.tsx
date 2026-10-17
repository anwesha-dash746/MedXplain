"use client";

import type { PredictResponse } from "@/lib/types";
import { isCalibrated } from "./PlainLanguage";

const BAND_COLOR: Record<string, string> = {
  low: "#0f9d76",
  borderline: "#c58a12",
  elevated: "#dd6b20",
  high: "#c53030",
};

const BAND_LABEL: Record<string, string> = {
  low: "Low",
  borderline: "Borderline",
  elevated: "Elevated",
  high: "High",
};

/**
 * Semicircular gauge for the risk probability.
 *
 * The threshold is drawn on the arc as a tick rather than left implicit. For a
 * rare outcome the operating point is nowhere near 50%, so a gauge without it
 * invites the reader to assume "under half the dial = safe", which is exactly
 * the wrong reading.
 */
export function RiskGauge({ result }: { result: PredictResponse }) {
  const { probability, decision_threshold: threshold, risk_band: band } = result;
  const color = BAND_COLOR[band] ?? "#64748b";
  // Without calibration evidence the arc is a ranking position, not a percentage.
  const calibrated = isCalibrated(result);

  const radius = 92;
  const stroke = 18;
  const cx = 110;
  const cy = 110;
  const circumference = Math.PI * radius;

  return (
    <div className="flex flex-col items-center">
      <svg width="220" height="128" viewBox="0 0 220 128" role="img" aria-label={calibrated ? `Risk ${(probability * 100).toFixed(1)} percent` : `${BAND_LABEL[band] ?? band} model score`}>
        <path
          d={arcPath(cx, cy, radius)}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth={stroke}
          strokeLinecap="round"
        />
        <path
          d={arcPath(cx, cy, radius)}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${circumference * probability} ${circumference}`}
          style={{ transition: "stroke-dasharray 600ms ease, stroke 300ms ease" }}
        />
        {/* decision threshold tick */}
        <line
          {...tickCoords(cx, cy, radius, stroke, threshold)}
          stroke="#0f172a"
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        <text x={cx} y={cy - 26} textAnchor="middle" className="fill-slate-900" style={{ fontSize: calibrated ? 34 : 24, fontWeight: 700 }}>
          {calibrated ? `${(probability * 100).toFixed(1)}%` : BAND_LABEL[band] ?? band}
        </text>
        <text x={cx} y={cy - 6} textAnchor="middle" className="fill-slate-500" style={{ fontSize: 11 }}>
          {calibrated ? "predicted risk" : "model score (not a %)"}
        </text>
      </svg>

      <div className="mt-1 flex items-center gap-2">
        <span
          className="chip text-white"
          style={{ backgroundColor: color }}
        >
          {BAND_LABEL[band] ?? band} risk
        </span>
        {calibrated && (
          <span className="text-xs text-slate-500">
            alert at {(threshold * 100).toFixed(1)}%
          </span>
        )}
      </div>

      <p className="mt-2 text-center text-xs text-slate-500">
        {result.predicted_label === 1
          ? "Above the model's alert level, so this patient is flagged."
          : "Below the model's alert level, so this patient is not flagged."}
        <br />
        The black tick on the dial marks the alert level.
      </p>
    </div>
  );
}

function arcPath(cx: number, cy: number, r: number): string {
  return `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;
}

function tickCoords(cx: number, cy: number, r: number, stroke: number, fraction: number) {
  const angle = Math.PI * (1 - Math.min(Math.max(fraction, 0), 1));
  const inner = r - stroke / 2 - 2;
  const outer = r + stroke / 2 + 2;
  return {
    x1: cx + inner * Math.cos(angle),
    y1: cy - inner * Math.sin(angle),
    x2: cx + outer * Math.cos(angle),
    y2: cy - outer * Math.sin(angle),
  };
}
