import { DataTable, type Column } from "./DataTable";
import { PropPlayerCell } from "./StarButton";
import type { CrossBookRow } from "../api/types";
import { fmtLine, fmtPct, statLabel } from "../lib/format";

// The "middle" chip is deliberately worded as a *candidate*, never
// "guaranteed" — a middle only wins both legs if the result lands in the gap.
function OpportunityBadge({ type }: { type: string | null }) {
  if (type === "middle_candidate") {
    return (
      <span
        className="tnum inline-flex h-6 items-center rounded-md border border-warn/40 bg-warn/10 px-2 text-caption font-medium text-warn"
        title="Middle candidate — wins both legs only if the result lands in the gap"
      >
        MIDDLE?
      </span>
    );
  }
  return (
    <span className="tnum inline-flex h-6 items-center rounded-md border border-line bg-surface-2 px-2 text-caption font-medium text-muted">
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
      render: (r) => <PropPlayerCell name={r.player_name} stat={r.stat_type} />,
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
      header: "Books",
      align: "right",
      render: (r) => <span className="tnum text-muted">{r.n_books}</span>,
      sortable: true,
      sortValue: (r) => r.n_books,
    },
    {
      key: "line_min",
      header: "Low",
      align: "right",
      render: (r) => <span className="tnum">{fmtLine(r.line_min)}</span>,
      sortable: true,
      sortValue: (r) => r.line_min,
    },
    {
      key: "line_max",
      header: "High",
      align: "right",
      render: (r) => <span className="tnum">{fmtLine(r.line_max)}</span>,
      sortable: true,
      sortValue: (r) => r.line_max,
    },
    {
      key: "line_gap",
      header: "Gap",
      align: "right",
      render: (r) => (
        <span className="tnum font-semibold text-fg">{fmtLine(r.line_gap)}</span>
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
          <span className="tnum ml-2 text-faint">@ {fmtLine(r.line_min)}</span>
        </span>
      ),
    },
    {
      key: "best_under_book",
      header: "Best UNDER",
      render: (r) => (
        <span className="text-muted">
          {r.best_under_book ?? "—"}
          <span className="tnum ml-2 text-faint">@ {fmtLine(r.line_max)}</span>
        </span>
      ),
    },
    {
      key: "consensus_mean",
      header: "Consensus",
      align: "right",
      render: (r) => (
        <span className="tnum text-muted">{fmtLine(r.consensus_mean)}</span>
      ),
      sortable: true,
      sortValue: (r) => r.consensus_mean,
    },
    {
      key: "p_range",
      header: "P(over) low → high",
      align: "right",
      render: (r) => (
        <span className="tnum text-faint">
          {fmtPct(r.p_over_at_line_min)}
          <span className="mx-1 text-faint" aria-hidden>→</span>
          {fmtPct(r.p_over_at_line_max)}
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
      rowActionLabel={(r) => `Open ${r.player_name} ${statLabel(r.stat_type)}`}
      maxHeight="calc(100vh - 220px)"
    />
  );
}
