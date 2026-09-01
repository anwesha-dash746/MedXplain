"use client";

import type { ExplanationMethod, MethodInfo, ModelInfo } from "@/lib/types";

const COST_STYLE: Record<string, string> = {
  fast: "bg-emerald-50 text-emerald-700 border-emerald-200",
  moderate: "bg-amber-50 text-amber-700 border-amber-200",
  slow: "bg-rose-50 text-rose-700 border-rose-200",
};

interface Props {
  models: ModelInfo[];
  methods: MethodInfo[];
  selectedModel: string;
  selectedMethod: ExplanationMethod;
  compareMode: boolean;
  compareMethods: ExplanationMethod[];
  onModelChange: (key: string) => void;
  onMethodChange: (key: ExplanationMethod) => void;
  onCompareToggle: (enabled: boolean) => void;
  onCompareMethodsChange: (methods: ExplanationMethod[]) => void;
  disabled: boolean;
}

export function Controls({
  models,
  methods,
  selectedModel,
  selectedMethod,
  compareMode,
  compareMethods,
  onModelChange,
  onMethodChange,
  onCompareToggle,
  onCompareMethodsChange,
  disabled,
}: Props) {
  const activeMethod = methods.find((m) => m.key === selectedMethod);
  const activeModel = models.find((m) => m.key === selectedModel);

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">Model &amp; explanation</h2>
      </div>

      <div className="space-y-4 p-4">
        <div>
          <label className="field-label" htmlFor="model-select">
            Fine-tuned model
          </label>
          <select
            id="model-select"
            className="select"
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
            disabled={disabled || models.length === 0}
          >
            {models.map((model) => (
              <option key={model.key} value={model.key} disabled={!model.available}>
                {model.display_name}
                {model.test_auroc != null ? ` — AUROC ${model.test_auroc.toFixed(3)}` : ""}
              </option>
            ))}
          </select>
          {activeModel?.hf_id && (
            <p className="mt-1 truncate font-mono text-[11px] text-slate-400">{activeModel.hf_id}</p>
          )}
        </div>

        <div>
          <div className="mb-1 flex items-center justify-between">
            <span className="field-label !mb-0">Explanation method</span>
            <label className="flex cursor-pointer items-center gap-1.5 text-xs text-slate-600">
              <input
                type="checkbox"
                className="h-3.5 w-3.5 rounded border-slate-300 text-clinical-600 focus:ring-clinical-500"
                checked={compareMode}
                onChange={(e) => onCompareToggle(e.target.checked)}
                disabled={disabled}
              />
              Compare methods
            </label>
          </div>

          {compareMode ? (
            <div className="space-y-1.5">
              {methods.map((method) => (
                <label
                  key={method.key}
                  className="flex cursor-pointer items-start gap-2 rounded-lg border border-slate-200 px-2.5 py-2 hover:bg-slate-50"
                >
                  <input
                    type="checkbox"
                    className="mt-0.5 h-3.5 w-3.5 rounded border-slate-300 text-clinical-600 focus:ring-clinical-500"
                    checked={compareMethods.includes(method.key)}
                    onChange={(e) => {
                      const next = e.target.checked
                        ? [...compareMethods, method.key]
                        : compareMethods.filter((m) => m !== method.key);
                      onCompareMethodsChange(next);
                    }}
                    disabled={disabled}
                  />
                  <span className="flex-1 text-xs">
                    <span className="font-medium text-slate-700">{method.display_name}</span>
                    <CostChip cost={method.cost} typicalMs={method.typical_ms} />
                  </span>
                </label>
              ))}
              {compareMethods.length === 0 && (
                <p className="text-xs text-amber-600">Select at least one method.</p>
              )}
            </div>
          ) : (
            <>
              <select
                className="select"
                value={selectedMethod}
                onChange={(e) => onMethodChange(e.target.value as ExplanationMethod)}
                disabled={disabled}
              >
                {methods.map((method) => (
                  <option key={method.key} value={method.key}>
                    {method.display_name}
                  </option>
                ))}
              </select>
              {activeMethod && (
                <div className="mt-2 rounded-lg bg-slate-50 p-2.5">
                  <div className="mb-1 flex items-center gap-2">
                    <CostChip cost={activeMethod.cost} typicalMs={activeMethod.typical_ms} />
                    <span
                      className={`chip border ${
                        activeMethod.signed
                          ? "border-slate-300 bg-white text-slate-600"
                          : "border-violet-200 bg-violet-50 text-violet-700"
                      }`}
                    >
                      {activeMethod.signed ? "signed" : "unsigned"}
                    </span>
                  </div>
                  <p className="text-[11px] leading-relaxed text-slate-600">
                    {activeMethod.description}
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function CostChip({ cost, typicalMs }: { cost: string; typicalMs?: number | null }) {
  const label =
    typicalMs == null
      ? cost
      : typicalMs >= 1000
        ? `~${(typicalMs / 1000).toFixed(0)}s`
        : `~${typicalMs}ms`;
  return (
    <span className={`chip ml-1.5 border ${COST_STYLE[cost] ?? COST_STYLE.moderate}`}>{label}</span>
  );
}
