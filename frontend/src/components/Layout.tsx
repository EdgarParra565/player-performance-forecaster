import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { NavLink, Outlet, useRouteError } from "react-router-dom";
import { useHealth } from "../api/hooks";
import { AccessGate } from "./AccessGate";
import { DbNotice } from "./DbNotice";
import { isDbNotMounted } from "./Loading";
import { fmtAgo, fmtDateShort, parseApiDate } from "../lib/format";

interface NavItem {
  to: string;
  label: string;
  short: string; // bottom-bar label
  end: boolean;
  icon: string;
}

const NAV: NavItem[] = [
  { to: "/", label: "Slate", short: "Slate", end: true, icon: "M3 13h4V7H3v6zm6 0h4V3H9v10zm6 0h4v-4h-4v4z" },
  { to: "/player", label: "Player", short: "Player", end: false, icon: "M10 10a3 3 0 100-6 3 3 0 000 6zm-6 7a6 6 0 0112 0" },
  { to: "/edges", label: "Edge Scanner", short: "Edges", end: false, icon: "M3 15l4-5 3 3 7-8" },
  { to: "/cross-book", label: "Cross-book", short: "Cross", end: false, icon: "M4 7h12M4 7l3-3M4 7l3 3M16 13H4m12 0l-3-3m3 3l-3 3" },
  { to: "/teams", label: "Team Charts", short: "Teams", end: false, icon: "M4 16V9m4 7V5m4 11v-5m4 5V8" },
  { to: "/parlay", label: "Parlay", short: "Parlay", end: false, icon: "M5 5h10M5 10h10M5 15h6m4 0h.01" },
  { to: "/paper-trades", label: "Paper Trades", short: "Trades", end: false, icon: "M6 3h8l3 3v11H3V3h3m0 7l2.5 2.5L14 8" },
  { to: "/watchlist", label: "Watchlist", short: "Watch", end: false, icon: "M10 2.5l2.3 4.7 5.2.8-3.8 3.7.9 5.1L10 14.4l-4.6 2.4.9-5.1L2.5 8l5.2-.8L10 2.5z" },
];

// Phone bottom bar: the four most-used views + "More" (opens the drawer).
const PRIMARY = ["/", "/player", "/edges", "/watchlist"];

export function Icon({ d, className = "h-4 w-4" }: { d: string; className?: string }) {
  return (
    <svg viewBox="0 0 20 20" className={`${className} shrink-0`} aria-hidden>
      <path d={d} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-line-strong bg-surface-2 shadow-[var(--shadow-card)]">
        <svg viewBox="0 0 20 20" className="h-4 w-4 text-accent" aria-hidden>
          <path d="M3 14l4-5 3 2.5L17 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      <div className={`leading-tight ${compact ? "sr-only" : ""}`}>
        <div className="text-title font-semibold tracking-tight text-fg">NBA Props</div>
        <div className="text-caption text-faint">Terminal</div>
      </div>
    </div>
  );
}

function navClass(isActive: boolean) {
  return `flex items-center gap-3 rounded-lg px-3 py-2 text-body font-medium transition-colors pointer-coarse:min-h-11 ${
    isActive
      ? "bg-surface-3 text-fg shadow-[0_1px_0_0_rgb(255_255_255/0.05)_inset]"
      : "text-muted hover:bg-surface-2 hover:text-fg"
  }`;
}

// Freshness pill: green dot only when a scrape landed in the last 6h (that is
// a statement about data, not decoration). Compact form drops the age text.
function FreshnessPill({ freshest, noDb = false }: { freshest: string | null; noDb?: boolean }) {
  const d = parseApiDate(freshest);
  const live = d ? Date.now() - d.getTime() < 6 * 3_600_000 : false;
  if (noDb) {
    // Never say "Offseason" when there is no database at all.
    return (
      <span className="inline-flex h-7 items-center gap-2 rounded-full border border-neg-dim/60 bg-neg-soft px-3 text-caption whitespace-nowrap text-neg">
        <span className="h-1.5 w-1.5 rounded-full bg-neg" aria-hidden />
        No database
      </span>
    );
  }
  return (
    <span className="inline-flex h-7 items-center gap-2 rounded-full border border-line bg-surface-1 px-3 text-caption whitespace-nowrap text-muted">
      <span className={`h-1.5 w-1.5 rounded-full ${live ? "bg-pos" : "bg-faint"}`} aria-hidden />
      {live ? "Lines live" : "Offseason"}
      <span className="tnum hidden text-faint sm:inline">· scraped {fmtAgo(freshest)}</span>
    </span>
  );
}

function NavList({ onNavigate, iconOnly = false }: { onNavigate?: () => void; iconOnly?: boolean }) {
  return (
    <nav aria-label="Primary" className={`flex flex-1 flex-col gap-1 ${iconOnly ? "items-center px-2" : "px-3"}`}>
      {NAV.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.end}
          onClick={onNavigate}
          title={iconOnly ? item.label : undefined}
          aria-label={iconOnly ? item.label : undefined}
          className={({ isActive }) => `${navClass(isActive)} ${iconOnly ? "h-11 w-11 justify-center px-0" : ""}`}
        >
          {({ isActive }) => (
            <>
              <span className={isActive ? "text-accent" : "text-faint"}>
                <Icon d={item.icon} />
              </span>
              {!iconOnly && item.label}
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );
}

// Slide-in navigation drawer for phones (opened from the header or "More").
function Drawer({ open, onClose, footer }: { open: boolean; onClose: () => void; footer: ReactNode }) {
  const panel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    panel.current?.querySelector<HTMLElement>("a,button")?.focus();
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden"; // no background scroll under the sheet
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-40 md:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
      <button type="button" aria-label="Close menu" onClick={onClose} className="absolute inset-0 bg-base/70 backdrop-blur-sm" />
      <div ref={panel} className="absolute inset-y-0 left-0 flex w-72 max-w-[85vw] flex-col border-r border-line bg-surface-1 pt-[env(safe-area-inset-top)] shadow-[var(--shadow-pop)]">
        <div className="flex items-center justify-between px-5 py-4">
          <BrandMark />
          <button type="button" onClick={onClose} aria-label="Close menu" className="inline-flex h-11 w-11 items-center justify-center rounded-lg text-muted hover:bg-surface-2 hover:text-fg">
            <Icon d="M5 5l10 10M15 5L5 15" />
          </button>
        </div>
        <NavList onNavigate={onClose} />
        {footer}
      </div>
    </div>
  );
}

function BottomNav({ onMore }: { onMore: () => void }) {
  const items = NAV.filter((n) => PRIMARY.includes(n.to));
  return (
    <nav
      aria-label="Quick navigation"
      className="fixed inset-x-0 bottom-0 z-30 grid grid-cols-5 border-t border-line bg-surface-1/95 pb-[env(safe-area-inset-bottom)] backdrop-blur md:hidden"
    >
      {items.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.end}
          className={({ isActive }) =>
            `flex h-14 flex-col items-center justify-center gap-1 text-caption font-medium ${isActive ? "text-fg" : "text-faint"}`
          }
        >
          {({ isActive }) => (
            <>
              <span className={isActive ? "text-accent" : ""}>
                <Icon d={item.icon} className="h-5 w-5" />
              </span>
              {item.short}
            </>
          )}
        </NavLink>
      ))}
      <button type="button" onClick={onMore} className="flex h-14 flex-col items-center justify-center gap-1 text-caption font-medium text-faint">
        <Icon d="M4 6h12M4 10h12M4 14h12" className="h-5 w-5" />
        More
      </button>
    </nav>
  );
}

export function Layout() {
  const health = useHealth();
  const online = health.data?.db_exists ?? false;
  const freshest = health.data?.freshest_scrape_utc ?? null;
  const lastGame = health.data?.last_game_date ?? null;
  const [drawer, setDrawer] = useState(false);
  const closeDrawer = useCallback(() => setDrawer(false), []);

  const status = (
    <div className="border-t border-line px-5 py-4">
      <div className="flex items-center gap-2 text-caption text-muted">
        <span className={`h-1.5 w-1.5 rounded-full ${online ? "bg-accent" : "bg-neg"}`} aria-hidden />
        {online ? "Database online" : "Database offline"}
      </div>
      <div className="tnum mt-1 text-caption text-faint">v{health.data?.version ?? "—"} · read-only</div>
    </div>
  );

  return (
    <div className="flex min-h-screen text-fg">
      <AccessGate required={health.data?.access_code_required ?? false} />

      {/* Tablet (md): icon rail. Desktop (lg): full rail. Phone: drawer + bottom bar. */}
      <aside className="hidden w-16 shrink-0 border-r border-line bg-surface-1/80 md:block lg:w-60">
        <div className="sticky top-0 flex h-screen flex-col">
          <div className="flex justify-center px-2 py-5 lg:justify-start lg:px-5">
            <span className="lg:hidden">
              <BrandMark compact />
            </span>
            <span className="hidden lg:block">
              <BrandMark />
            </span>
          </div>
          <div className="flex flex-1 flex-col lg:hidden">
            <NavList iconOnly />
          </div>
          <div className="hidden flex-1 flex-col lg:flex">
            <NavList />
          </div>
          <div className="hidden lg:block">{status}</div>
        </div>
      </aside>

      <Drawer open={drawer} onClose={closeDrawer} footer={status} />

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 border-b border-line bg-base/85 pt-[env(safe-area-inset-top)] backdrop-blur">
          <div className="flex h-14 items-center justify-between gap-3 px-4 md:px-8">
            <div className="flex min-w-0 items-center gap-2 md:hidden">
              <button
                type="button"
                onClick={() => setDrawer(true)}
                aria-label="Open menu"
                aria-expanded={drawer}
                className="-ml-2 inline-flex h-11 w-11 items-center justify-center rounded-lg text-muted hover:bg-surface-2 hover:text-fg"
              >
                <Icon d="M4 6h12M4 10h12M4 14h12" className="h-5 w-5" />
              </button>
              <BrandMark />
            </div>
            <div className="hidden items-center gap-2 text-caption text-faint md:flex">
              Model-vs-market analytics · not betting advice
            </div>
            <div className="flex items-center gap-3">
              <span className="hidden text-caption text-faint lg:inline">
                last game <span className="tnum text-muted">{fmtDateShort(lastGame)}</span>
              </span>
              <FreshnessPill freshest={freshest} noDb={isDbNotMounted(health.error)} />
            </div>
          </div>
        </header>

        {/* pb clears the fixed bottom bar on phones */}
        <main className="flex-1 px-4 pt-6 pb-28 md:px-8 md:py-8">
          <DbNotice error={health.error} />
          <Outlet />
        </main>
      </div>

      <BottomNav onMore={() => setDrawer(true)} />
    </div>
  );
}

// Router errorElement: a render crash in any view shows this instead of the
// router's default "Unexpected Application Error" page.
export function RouteError() {
  const error = useRouteError();
  const message = error instanceof Error ? error.message : "Unknown error";
  return (
    <div className="flex min-h-screen items-center justify-center p-8 text-fg">
      <div className="panel max-w-md p-8 text-center">
        <div className="text-title font-semibold">Something broke on this page.</div>
        <p className="mt-2 text-body text-muted">
          The view hit an unexpected error. Reload, or head back to the slate.
        </p>
        <pre className="tnum mt-4 overflow-auto rounded-md bg-surface-2 p-3 text-left text-caption text-faint">
          {message}
        </pre>
        <a href="/" className="mt-6 inline-flex h-8 items-center rounded-md border border-line-strong bg-surface-2 px-4 text-label font-medium hover:bg-surface-3">
          Back to slate
        </a>
      </div>
    </div>
  );
}
