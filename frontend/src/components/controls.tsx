import type { ReactNode } from "react";

export function Chip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`tnum rounded border px-2 py-0.5 text-[11px] transition-colors ${
        active
          ? "border-pos-dim bg-pos-soft text-pos"
          : "border-line bg-panel text-muted hover:border-line-strong hover:text-fg"
      }`}
    >
      {children}
    </button>
  );
}

export interface SegmentedOption {
  key: string;
  label: string;
}

export function Segmented({
  options,
  value,
  onChange,
}: {
  options: SegmentedOption[];
  value: string;
  onChange: (key: string) => void;
}) {
  return (
    <div className="flex gap-1 rounded border border-line bg-panel p-1">
      {options.map((o) => (
        <button
          key={o.key}
          onClick={() => onChange(o.key)}
          className={`tnum rounded px-2.5 py-1 text-[11px] transition-colors ${
            value === o.key ? "bg-panel-3 text-fg" : "text-muted hover:text-fg"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function NumberField({
  label,
  value,
  onChange,
  step = 1,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <label className="flex items-center gap-2 text-[11px] text-faint">
      {label}
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isNaN(n)) return;
          let v = n;
          if (min !== undefined) v = Math.max(min, v);
          if (max !== undefined) v = Math.min(max, v);
          onChange(v);
        }}
        className="tnum w-16 rounded border border-line bg-panel-2 px-2 py-1 text-fg focus:border-line-strong focus:outline-none"
      />
    </label>
  );
}
