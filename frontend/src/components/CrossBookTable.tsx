import { DataTable, type Column } from "./DataTable";
import type { CrossBookRow } from "../api/types";
import { fmtNum, fmtPct, statLabel } from "../lib/format";

// The "middle" chip is deliberately worded as a *candidate*, never
// "guaranteed" — a middle only wins both legs if the result lands in the gap.
function OpportunityBadge({ type }: { type: string | null }) {
  if (type === "middle_candidate") {
    return (
      <span className="tnum rounded border border-warn/50 px-1.5 py-0.5 text-[10px] text-warn">
        MIDDLE?
      </span>
    );
  }
  return (
    <span className="tnum rounded border border-line px-1.5 py-0.5 text-[10px] text-muted">
      SHOP
    </span>
  );
}

export function CrossBookTable({
  rows,
  onRowClick,
}: {
  rows: CrossBookRow[];
  onRowClick?: (row: CrossBookRow) => void;
}) {
  const columns: Column<CrossBookRow>[] = [
    {
      key: "player_name",
      header: "Player",
      render: (r) => <span className="text-fg">{r.player_name}</span>,
      sortable: true,
      sortValue: (r) => r.player_name,
    },
    {
      key: "stat_type",
      header: "Stat",
      render: (r) => (
        <span className="tnum text-muted">{statLabel(r.stat_type)}</span>
      ),
      sortable: true,
      sortValue: (r) => r.stat_type,
    },
    {
      key: "n_books",
      header: "Bk",
      align: "right",
      render: (r) => <span className="tnum text-muted">{r.n_books}</span>,
      sortable: true,
      sortValue: (r) => r.n_books,
    },
    {
      key: "line_min",
      header: "Low",
      align: "right",
      render: (r) => <span className="tnum">{fmtNum(r.line_min)}</span>,
      sortable: true,
      sortValue: (r) => r.line_min,
    },
    {
      key: "line_max",
      header: "High",
      align: "right",
      render: (r) => <span className="tnum">{fmtNum(r.line_max)}</span>,
      sortable: true,
      sortValue: (r) => r.line_max,
    },
    {
      key: "line_gap",
      header: "Gap",
      align: "right",
      render: (r) => (
        <span className="tnum font-semibold text-fg">{fmtNum(r.line_gap)}</span>
      ),
      sortable: true,
      sortValue: (r) => r.line_gap,
    },
    {
      key: "best_over_book",
      header: "Best OVER",
      render: (r) => (
        <span className="text-muted">
          {r.best_over_book ?? "—"}
          <span className="tnum ml-1 text-faint">@{fmtNum(r.line_min)}</span>
        </span>
      ),
    },
    {
      key: "best_under_book",
      header: "Best UNDER",
      render: (r) => (
        <span className="text-muted">
          {r.best_under_book ?? "—"}
          <span className="tnum ml-1 text-faint">@{fmtNum(r.line_max)}</span>
        </span>
      ),
    },
    {
      key: "consensus_mean",
      header: "Cons",
      align: "right",
      render: (r) => (
        <span className="tnum text-muted">{fmtNum(r.consensus_mean)}</span>
      ),
      sortable: true,
      sortValue: (r) => r.consensus_mean,
    },
    {
      key: "p_range",
      header: "P(o) low→high",
      align: "right",
      render: (r) => (
        <span className="tnum text-faint">
          {fmtPct(r.p_over_at_line_min, 0)}
          <span className="mx-1 text-line-strong">→</span>
          {fmtPct(r.p_over_at_line_max, 0)}
        </span>
      ),
    },
    {
      key: "opportunity_type",
      header: "Type",
      render: (r) => <OpportunityBadge type={r.opportunity_type} />,
      sortable: true,
      sortValue: (r) => r.opportunity_type,
    },
  ];

  return (
    <DataTable
      columns={columns}
      rows={rows}
      rowKey={(r) => `${r.player_name}-${r.stat_type}`}
      initialSort={{ key: "line_gap", dir: "desc" }}
      onRowClick={onRowClick}
      maxHeight="calc(100vh - 470px)"
    />
  );
}
