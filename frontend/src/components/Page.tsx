import type { ReactNode } from "react";

// Page shell: every view gets the same title block, action slot, and 24px
// vertical rhythm, so the five views read as one product.
export function Page({
  title,
  description,
  actions,
  eyebrow,
  children,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  eyebrow?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="mx-auto w-full max-w-[1480px] space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          {eyebrow && <div className="eyebrow mb-1">{eyebrow}</div>}
          <h1 className="text-heading font-semibold text-fg">{title}</h1>
          {description && (
            <p className="mt-1 max-w-3xl text-body text-muted">{description}</p>
          )}
        </div>
        {actions && <div className="flex w-full flex-wrap items-center gap-3 sm:w-auto">{actions}</div>}
      </header>
      {children}
    </div>
  );
}

// Card: titled container at elevation 1. `tone="pos"` is reserved for
// surfaces whose whole meaning is +EV (the true-arb section).
export function Card({
  title,
  meta,
  actions,
  tone = "default",
  flush = false,
  className = "",
  children,
}: {
  title?: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  tone?: "default" | "pos";
  flush?: boolean;
  className?: string;
  children: ReactNode;
}) {
  const shell =
    tone === "pos"
      ? "rounded-[var(--radius-card)] border border-pos-dim/50 bg-pos-soft/40 shadow-[var(--shadow-card)]"
      : "panel";
  return (
    <section className={`${shell} overflow-hidden ${className}`}>
      {(title || meta || actions) && (
        <div
          className={`flex min-h-12 flex-wrap items-center justify-between gap-2 border-b px-4 py-3 ${
            tone === "pos" ? "border-pos-dim/40" : "border-line"
          }`}
        >
          {title && (
            <h2
              className={`text-title font-semibold ${tone === "pos" ? "text-pos" : "text-fg"}`}
            >
              {title}
            </h2>
          )}
          <div className="flex items-center gap-3">
            {meta && <span className="text-label text-faint">{meta}</span>}
            {actions}
          </div>
        </div>
      )}
      <div className={flush ? "" : "p-4"}>{children}</div>
    </section>
  );
}

// Grid for KPI rows.
export function KpiGrid({ cols = 4, children }: { cols?: 4 | 5 | 6; children: ReactNode }) {
  // 2-up on phones; tablets get 3/4 columns before the full desktop row.
  const cls = {
    4: "md:grid-cols-4",
    5: "md:grid-cols-3 lg:grid-cols-5",
    6: "md:grid-cols-3 lg:grid-cols-6",
  }[cols];
  return <div className={`grid grid-cols-2 gap-3 sm:gap-4 ${cls}`}>{children}</div>;
}
