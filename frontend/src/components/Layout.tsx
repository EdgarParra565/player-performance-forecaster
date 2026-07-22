import { NavLink, Outlet } from "react-router-dom";
import { useHealth } from "../api/hooks";
import { fmtAgo, fmtDateShort } from "../lib/format";

const NAV = [
  { to: "/", label: "Slate", end: true },
  { to: "/player", label: "Player", end: false },
  { to: "/edges", label: "Edge Scanner", end: false },
];

export function Layout() {
  const health = useHealth();
  const online = health.data?.db_exists ?? false;
  const freshest = health.data?.freshest_scrape_utc ?? null;
  const lastGame = health.data?.last_game_date ?? null;

  return (
    <div className="flex min-h-screen bg-base text-fg">
      {/* Left rail */}
      <aside className="fixed inset-y-0 left-0 flex w-52 flex-col border-r border-line bg-panel">
        <div className="border-b border-line px-4 py-4">
          <div className="tnum text-[13px] font-bold tracking-wide text-fg">
            NBA<span className="text-pos">·</span>PROPS
          </div>
          <div className="eyebrow mt-0.5">Terminal</div>
        </div>

        <nav className="flex flex-1 flex-col gap-0.5 p-2">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `flex items-center gap-2 rounded px-3 py-2 text-[13px] transition-colors ${
                  isActive
                    ? "bg-panel-3 text-fg"
                    : "text-muted hover:bg-panel-2 hover:text-fg"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <span
                    className={`h-1 w-1 rounded-full ${
                      isActive ? "bg-pos" : "bg-faint"
                    }`}
                  />
                  {item.label}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="border-t border-line px-4 py-3">
          <div className="flex items-center gap-2">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                online ? "bg-pos" : "bg-neg"
              }`}
            />
            <span className="eyebrow">{online ? "DB online" : "DB offline"}</span>
          </div>
          <div className="tnum mt-1.5 text-[10px] text-faint">
            v{health.data?.version ?? "—"}
          </div>
        </div>
      </aside>

      {/* Main */}
      <div className="ml-52 flex min-h-screen flex-1 flex-col">
        <header className="sticky top-0 z-20 flex items-center justify-between border-b border-line bg-base/90 px-6 py-2.5 backdrop-blur">
          <div className="flex items-center gap-2 text-xs text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-pos/70" />
            <span className="eyebrow">Read-only · local</span>
          </div>
          <div className="flex items-center gap-5 text-[11px]">
            <span className="text-faint">
              last game{" "}
              <span className="tnum text-muted">{fmtDateShort(lastGame)}</span>
            </span>
            <span className="text-faint">
              last scrape{" "}
              <span className="tnum text-muted">{fmtAgo(freshest)}</span>
            </span>
          </div>
        </header>

        <main className="flex-1 p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
