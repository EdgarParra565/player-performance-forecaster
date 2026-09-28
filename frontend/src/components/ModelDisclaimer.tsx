import type { ReactNode } from "react";

// The standing honest-labeling convention for anything that looks like a
// betting recommendation. Warn-toned (not green/red): it's a caveat, not EV.
export function ModelDisclaimer({ children }: { children?: ReactNode }) {
  return (
    <div
      role="note"
      className="flex items-start gap-3 rounded-lg border border-warn/30 bg-warn/5 px-4 py-3 text-body text-muted"
    >
      <svg viewBox="0 0 20 20" aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-warn">
        <path d="M10 3l8 14H2L10 3z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
        <path d="M10 8v4m0 2.5v.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
      <div>
        <span className="font-semibold text-warn">Model estimate — not validated for real money.</span>{" "}
        WS10 calibration gates are unpassed.
        {children && <> {children}</>}
      </div>
    </div>
  );
}
