import type { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  // Polarity accent on the value — only for values that ARE +EV / -EV.
  tone?: "default" | "pos" | "neg" | "warn";
  accent?: boolean;
}

const TONE: Record<string, string> = {
  default: "text-fg",
  pos: "text-pos",
  neg: "text-neg",
  warn: "text-warn",
};

// The KPI tile: small label, large tabular figure, quiet caption.
export function StatCard({ label, value, sub, tone = "default", accent }: StatCardProps) {
  return (
    <div className="panel relative overflow-hidden px-4 py-4">
      {accent && (
        <div
          className="absolute inset-x-0 top-0 h-px bg-linear-to-r from-pos/0 via-pos/80 to-pos/0"
          aria-hidden
        />
      )}
      <div className="text-label font-medium text-muted">{label}</div>
      <div className={`num mt-2 truncate text-kpi font-semibold ${TONE[tone]}`}>{value}</div>
      {sub && <div className="mt-1 truncate text-caption text-faint">{sub}</div>}
    </div>
  );
}

// Tone for a signed KPI (EV, CLV, units): green above zero, red below.
export function signedTone(v: number | null | undefined): "default" | "pos" | "neg" {
  if (v == null || !Number.isFinite(v) || Math.abs(v) < 1e-9) return "default";
  return v > 0 ? "pos" : "neg";
}
