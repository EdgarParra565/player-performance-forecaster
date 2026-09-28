import { useEffect, useState, type ReactNode } from "react";

// Shared form controls. One visual language: 32px tall, 6px radius, surface-2
// fill, accent (not polarity) for selection, visible focus rings.

export const MODEL_MODES = [
  { key: "chart_mean", label: "Chart mean" },
  { key: "rolling", label: "Rolling" },
  { key: "full", label: "Full (beta)" },
];

// pointer-coarse: 44px touch targets on phones/tablets; desktop stays dense.
const FIELD =
  "h-8 pointer-coarse:h-11 rounded-[var(--radius-control)] border border-line bg-surface-2 px-2 text-body text-fg " +
  "transition-colors hover:border-line-strong focus:border-accent/70 focus:outline-none";

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
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={`tnum h-7 rounded-full border px-3 text-caption transition-colors pointer-coarse:h-11 pointer-coarse:px-4 ${
        active
          ? "border-accent/60 bg-accent-soft text-fg"
          : "border-line bg-surface-2 text-muted hover:border-line-strong hover:text-fg"
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
  label,
}: {
  options: SegmentedOption[];
  value: string;
  onChange: (key: string) => void;
  label?: string;
}) {
  return (
    <div
      role="group"
      aria-label={label}
      className="inline-flex flex-wrap gap-1 rounded-lg border border-line bg-surface-1 p-1"
    >
      {options.map((o) => {
        const active = value === o.key;
        return (
          <button
            type="button"
            key={o.key}
            aria-pressed={active}
            onClick={() => onChange(o.key)}
            className={`h-7 rounded-md px-3 text-label font-medium transition-colors pointer-coarse:h-11 pointer-coarse:min-w-11 ${
              active
                ? "bg-surface-3 text-fg shadow-[0_1px_0_0_rgb(255_255_255/0.06)_inset]"
                : "text-muted hover:bg-surface-2 hover:text-fg"
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

// Labelled numeric input. Keeps a local draft string so the field can be
// cleared / typed through intermediate values; clamps + commits on blur or
// Enter (clamping per keystroke made "12" impossible to type with min=2).
export function NumberField({
  label,
  value,
  onChange,
  step = 1,
  min,
  max,
  width = "w-20",
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  step?: number;
  min?: number;
  max?: number;
  width?: string;
}) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);

  function commit() {
    const n = Number(draft);
    if (draft.trim() === "" || !Number.isFinite(n)) {
      setDraft(String(value));
      return;
    }
    let v = n;
    if (min !== undefined) v = Math.max(min, v);
    if (max !== undefined) v = Math.min(max, v);
    setDraft(String(v));
    if (v !== value) onChange(v);
  }

  return (
    <label className="flex items-center gap-2 text-label text-muted">
      {label}
      <input
        type="number"
        inputMode="decimal"
        value={draft}
        step={step}
        min={min}
        max={max}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
        }}
        className={`tnum ${FIELD} ${width}`}
      />
    </label>
  );
}

export function Select({
  label,
  value,
  onChange,
  options,
  hideLabel,
  format,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
  hideLabel?: boolean;
  // Display text per option (value stays the raw key), e.g. statLabel.
  format?: (v: string) => string;
}) {
  return (
    <label className="flex items-center gap-2 text-label text-muted">
      <span className={hideLabel ? "sr-only" : ""}>{label}</span>
      <span className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={`tnum ${FIELD} appearance-none pr-8 pl-3 font-medium`}
        >
          {options.map((o) => (
            <option key={o} value={o}>
              {format ? format(o) : o}
            </option>
          ))}
        </select>
        <svg
          viewBox="0 0 12 12"
          aria-hidden
          className="pointer-events-none absolute top-1/2 right-2.5 h-3 w-3 -translate-y-1/2 text-faint"
        >
          <path d="M3 4.5l3 3 3-3" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      </span>
    </label>
  );
}

export function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-label text-muted select-none pointer-coarse:min-h-11">
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="peer sr-only"
      />
      <span
        aria-hidden
        className={`relative h-4 w-7 rounded-full border transition-colors peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent ${
          checked ? "border-accent/60 bg-accent/40" : "border-line-strong bg-surface-2"
        }`}
      >
        <span
          className={`absolute top-0.5 h-2.5 w-2.5 rounded-full bg-fg transition-transform ${
            checked ? "translate-x-3.5" : "translate-x-0.5"
          }`}
        />
      </span>
      {label}
    </label>
  );
}

export function Button({
  onClick,
  disabled,
  children,
  title,
}: {
  onClick: () => void;
  disabled?: boolean;
  children: ReactNode;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className="inline-flex h-8 items-center gap-2 rounded-[var(--radius-control)] border border-line-strong bg-surface-2 px-3 text-label font-medium text-fg transition-colors enabled:hover:bg-surface-3 disabled:cursor-not-allowed disabled:opacity-40 pointer-coarse:h-11 pointer-coarse:px-4"
    >
      {children}
    </button>
  );
}

// Filter row: fixed-width label column + wrapping content.
export function FilterRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-wrap items-start gap-3">
      <span className="eyebrow flex h-7 w-full shrink-0 items-center sm:w-14 pointer-coarse:h-11">{label}</span>
      <div className="flex flex-1 flex-wrap gap-2">{children}</div>
    </div>
  );
}

// Debounced mirror of a fast-changing value (search boxes, numeric filters)
// so each keystroke doesn't fire a full scan.
export function useDebounced<T>(value: T, ms = 300): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(t);
  }, [value, ms]);
  return v;
}
