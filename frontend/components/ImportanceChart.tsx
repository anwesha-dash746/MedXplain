"use client";

import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ExplanationPayload } from "@/lib/types";

interface Props {
  explanation: ExplanationPayload;
  topK?: number;
}

/**
 * Horizontal bar chart of the highest-contributing words.
 *
 * For signed methods the chart is diverging around zero and shows both the words
 * that pushed the risk up and those that pulled it down — the protective side is
 * clinically informative and is usually thrown away by systems that only show a
 * "top features" list. For unsigned methods only magnitude is shown.
 */
export function ImportanceChart({ explanation, topK = 12 }: Props) {
  const data = buildData(explanation, topK);

  if (data.length === 0) {
    return (
      <p className="px-5 py-8 text-center text-sm text-slate-500">
        No words scored above zero for this explanation.
      </p>
    );
  }

  const maxAbs = Math.max(...data.map((d) => Math.abs(d.score)), 1e-9);
  const domain: [number, number] = explanation.signed ? [-maxAbs * 1.1, maxAbs * 1.1] : [0, maxAbs * 1.1];
  const height = Math.max(220, data.length * 26 + 40);

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
          <XAxis
            type="number"
            domain={domain}
            tick={{ fontSize: 11, fill: "#64748b" }}
            tickFormatter={(v: number) => v.toFixed(2)}
            axisLine={{ stroke: "#e2e8f0" }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={140}
            tick={{ fontSize: 11, fill: "#334155", fontFamily: "ui-monospace, monospace" }}
            axisLine={false}
            tickLine={false}
            interval={0}
          />
          <Tooltip
            cursor={{ fill: "rgba(148,163,184,0.12)" }}
            contentStyle={{
              borderRadius: 8,
              border: "1px solid #e2e8f0",
              fontSize: 12,
              boxShadow: "0 4px 12px rgba(15,23,42,0.08)",
            }}
            formatter={(value: number) => [
              value.toFixed(4),
              explanation.signed
                ? value > 0
                  ? "raised model score"
                  : "lowered model score"
                : "importance",
            ]}
          />
          {explanation.signed && <ReferenceLine x={0} stroke="#94a3b8" strokeWidth={1} />}
          <Bar dataKey="score" radius={[3, 3, 3, 3]} isAnimationActive={false}>
            {data.map((entry, index) => (
              <Cell
                key={index}
                fill={
                  !explanation.signed
                    ? "#7c3aed"
                    : entry.score > 0
                      ? "#dc2626"
                      : "#2563eb"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

interface Row {
  label: string;
  score: number;
}

function buildData(explanation: ExplanationPayload, topK: number): Row[] {
  const positive = explanation.top_positive.slice(0, explanation.signed ? Math.ceil(topK / 2) : topK);
  const negative = explanation.signed ? explanation.top_negative.slice(0, Math.floor(topK / 2)) : [];

  const rows = [...positive, ...negative]
    .filter((w) => Math.abs(w.score) > 1e-9)
    // most positive at the top, most negative at the bottom; Recharts renders a
    // vertical layout bottom-up, so the array is reversed
    .sort((a, b) => a.score - b.score)
    .map((w) => ({ label: truncate(w.text, 22), score: w.score }));

  return rows;
}

function truncate(text: string, max: number): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max - 1)}…` : clean;
}
