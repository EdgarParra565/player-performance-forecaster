import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useEdges } from "../api/hooks";
import { Card, Page } from "../components/Page";
import { DataTable, type Column } from "../components/DataTable";
import { EmptyState } from "../components/EmptyState";
import { ErrorState, errorMessage } from "../components/Loading";
import { TableSkeleton } from "../components/Skeleton";
import { Delta, SideTag } from "../components/Delta";
import { ProbabilityBar } from "../components/ProbabilityBar";
import { Button } from "../components/controls";
import { StarButton } from "../components/StarButton";
import { DASH, fmtEv, fmtInt, fmtLine, fmtSignedPct, statLabel } from "../lib/format";
import { playerHref } from "../lib/rows";
import { decodeWatchlist, encodeWatchlist, itemKey, matchWatchlist, useWatchlist, type WatchItem, type WatchRow } from "../lib/watchlist";

function detailHref(i: WatchItem, stat?: string): string {
  const s = stat ?? i.stat ?? "points";
  return i.player_id ? `/player/${i.player_id}?stat=${encodeURIComponent(s)}` : playerHref(i.player_name, s);
}

const COLUMNS: Column<WatchRow>[] = [
  {
    key: "player",
    header: "Player",
    render: (r) => (
      <span className="-ml-2 inline-flex items-center gap-1">
        <StarButton item={r.item} />
        <Link
          to={detailHref(r.item, r.best?.stat_type)}
          className="font-medium text-fg underline-offset-4 hover:underline pointer-coarse:py-3"
        >
          {r.item.player_name}
        </Link>
      </span>
    ),
    sortable: true,
    sortValue: (r) => r.item.player_name,
  },
  {
    key: "edge",
    header: "Edge",
    align: "right",
    render: (r) => <Delta value={r.best?.model_edge} format={fmtSignedPct} strong />,
    sortable: true,
    sortValue: (r) => r.best?.model_edge ?? null,
  },
  {
    key: "what",
    header: "Watching",
    render: (r) => (
      <span className="tnum text-muted">{r.item.kind === "prop" ? statLabel(r.item.stat) : "All stats"}</span>
    ),
  },
  {
    key: "stat",
    header: "Best line",
    render: (r) =>
      r.best ? (
        <span className="tnum text-muted">
          {statLabel(r.best.stat_type)} <span className="text-fg">{fmtLine(r.best.book_line)}</span> · {r.best.book}
        </span>
      ) : (
        <span className="text-faint">Not on the board</span>
      ),
  },
  { key: "side", header: "Side", render: (r) => (r.best ? <SideTag side={r.best.best_side} /> : <span className="text-faint">{DASH}</span>) },
  {
    key: "p",
    header: "P(best)",
    align: "right",
    width: "150px",
    render: (r) =>
      r.best ? (
        <ProbabilityBar value={r.best.best_side === "under" ? r.best.p_under : r.best.p_over} />
      ) : (
        <span className="text-faint">{DASH}</span>
      ),
  },
  {
    key: "ev",
    header: "EV / unit",
    align: "right",
    render: (r) => <Delta value={r.best?.ev_best} format={fmtEv} />,
  },
  {
    key: "n",
    header: "Lines",
    align: "right",
    render: (r) => <span className="tnum text-faint">{fmtInt(r.nLines)}</span>,
  },
];

function SharedBanner({ incoming, onDone }: { incoming: WatchItem[]; onDone: () => void }) {
  const { merge, replace } = useWatchlist();
  return (
    <div role="region" aria-label="Shared watchlist" className="panel flex flex-wrap items-center justify-between gap-3 px-4 py-3">
      <div className="text-body text-muted">
        Someone shared a watchlist with <span className="font-semibold text-fg">{incoming.length}</span> item
        {incoming.length === 1 ? "" : "s"}:{" "}
        <span className="text-fg">
          {incoming.slice(0, 4).map((i) => (i.kind === "prop" ? `${i.player_name} ${statLabel(i.stat)}` : i.player_name)).join(", ")}
          {incoming.length > 4 ? "…" : ""}
        </span>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button onClick={() => { merge(incoming); onDone(); }}>Add to mine</Button>
        <Button onClick={() => { replace(incoming); onDone(); }}>Replace mine</Button>
        <Button onClick={onDone}>Dismiss</Button>
      </div>
    </div>
  );
}

export function Watchlist() {
  const { items, replace } = useWatchlist();
  const [search, setSearch] = useSearchParams();
  const incoming = useMemo(() => decodeWatchlist(search.get("w")), [search]);
  const [copied, setCopied] = useState<string | null>(null);

  // One full-model scan feeds every row (cached, heavy tier) — only when needed.
  const board = useEdges({ model_mode: "full", limit: 500 }, { enabled: items.length > 0 });
  const rows = useMemo(() => matchWatchlist(items, board.data?.rows ?? []), [items, board.data]);
  const onBoard = rows.filter((r) => r.best).length;

  function dismissShared() {
    setSearch((prev) => {
      const next = new URLSearchParams(prev);
      next.delete("w");
      return next;
    }, { replace: true });
  }

  async function share() {
    const url = `${window.location.origin}/watchlist?w=${encodeURIComponent(encodeWatchlist(items))}`;
    try {
      await navigator.clipboard.writeText(url);
      setCopied("Link copied");
    } catch {
      setCopied(url); // clipboard blocked: show the URL to copy by hand
    }
  }

  return (
    <Page
      title="Watchlist"
      description="Players and props you've starred, with the best current line and model edge on the board. Stored in this browser only; share it with a link."
      actions={
        items.length ? (
          <>
            <Button onClick={share} title="Copy a link to this watchlist">
              Share link
            </Button>
            <Button onClick={() => replace([])} title="Remove every item">
              Clear
            </Button>
          </>
        ) : undefined
      }
    >
      {incoming.length > 0 && <SharedBanner incoming={incoming} onDone={dismissShared} />}
      {copied && (
        <p role="status" className="text-label break-all text-muted">
          {copied}
        </p>
      )}

      {items.length === 0 ? (
        <Card>
          <EmptyState
            title="Nothing starred yet."
            hint="Tap the ☆ next to a player in Player Detail, or next to any prop in the Slate, Edge Scanner or Cross-book tables. Your watchlist stays in this browser."
          />
        </Card>
      ) : (
        <Card
          title="Starred"
          meta={board.data ? `${onBoard} of ${items.length} on the board · full model` : undefined}
          flush
        >
          {board.isError ? (
            <div className="p-4">
              <ErrorState message={errorMessage(board.error, "Couldn't load the board.")} />
            </div>
          ) : board.isLoading ? (
            <TableSkeleton rows={Math.min(items.length, 6)} cols={7} />
          ) : (
            <DataTable
              columns={COLUMNS}
              rows={rows}
              rowKey={(r) => itemKey(r.item)}
              initialSort={{ key: "edge", dir: "desc" }}
            />
          )}
        </Card>
      )}
    </Page>
  );
}
