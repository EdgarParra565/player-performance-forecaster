import type { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  // Optional polarity accent on the value (e.g. +EV green).
  tone?: "default" | "pos" | "neg" | "warn";
  accent?: boolean;
}

const TONE: Record<string, string> = {
  default: "text-fg",
  pos: "text-pos",
  neg: "text-neg",
  warn: "text-warn",
};

// The KPI tile. A left accent rule + eyebrow label + large mono value.
export function StatCard({ label, value, sub, tone = "default", accent }: StatCardProps) {
  return (
    <div className="panel relative overflow-hidden px-4 py-3">
      {accent && (
        <div className="absolute inset-y-0 left-0 w-0.5 bg-pos" aria-hidden />
      )}
      <div className="eyebrow">{label}</div>
      <div className={`tnum mt-1.5 text-2xl leading-none font-semibold ${TONE[tone]}`}>
        {value}
      </div>
      {sub && <div className="mt-1.5 text-[11px] text-faint">{sub}</div>}
    </div>
  );
}
