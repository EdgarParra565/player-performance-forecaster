import { useMemo, useState, type KeyboardEvent, type ReactNode } from "react";

export interface Column<T> {
  key: string;
  header: string;
  align?: "left" | "right" | "center";
  sortable?: boolean;
  // Value used for sorting (defaults to none / not sortable).
  sortValue?: (row: T) => number | string | null;
  render: (row: T) => ReactNode;
  width?: string;
  headerClassName?: string;
  cellClassName?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T, index: number) => string;
  initialSort?: { key: string; dir: "asc" | "desc" };
  onRowClick?: (row: T) => void;
  // Accessible description of what activating a row does ("Open player").
  rowActionLabel?: (row: T) => string;
  isActiveRow?: (row: T) => boolean;
  maxHeight?: string;
  // Pin the first column while the table scrolls horizontally (phones).
  stickyFirst?: boolean;
}

type Dir = "asc" | "desc";

const ALIGN: Record<string, string> = {
  left: "text-left",
  right: "text-right",
  center: "text-center",
};

// Sticky first column: opaque background matching the row state, plus a soft
// right edge so scrolled content visibly slides under it.
const STICKY_BASE = "sticky left-0 z-[1] shadow-[1px_0_0_0_var(--color-line),6px_0_8px_-6px_rgb(0_0_0/0.6)]";
const STICKY_HEAD = `${STICKY_BASE} z-[2] bg-surface-2`;
const STICKY_CELL = `${STICKY_BASE} bg-surface-1 group-hover:bg-surface-2 group-focus-visible:bg-surface-3`;
const STICKY_CELL_ACTIVE = `${STICKY_BASE} bg-surface-3`;

const JUSTIFY: Record<string, string> = {
  left: "justify-start",
  right: "justify-end",
  center: "justify-center",
};

// Dense, information-first table with a sticky header and click-to-sort. Zebra
// striping is intentionally omitted; hairline row borders + hover carry the
// scanning burden without visual noise. Clickable rows are keyboard-reachable
// (Tab + Enter/Space) and sortable headers are real buttons with aria-sort.
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  initialSort,
  onRowClick,
  rowActionLabel,
  isActiveRow,
  maxHeight,
  stickyFirst = true,
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(initialSort?.key ?? null);
  const [dir, setDir] = useState<Dir>(initialSort?.dir ?? "desc");

  const sorted = useMemo(() => {
    if (!sortKey) return rows;
    const col = columns.find((c) => c.key === sortKey);
    const getValue = col?.sortValue;
    if (!getValue) return rows;
    const factor = dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const av = getValue(a);
      const bv = getValue(b);
      // Nulls always sink to the bottom regardless of direction.
      const aNull = av === null || av === undefined;
      const bNull = bv === null || bv === undefined;
      if (aNull && bNull) return 0;
      if (aNull) return 1;
      if (bNull) return -1;
      if (typeof av === "number" && typeof bv === "number") {
        return (av - bv) * factor;
      }
      return String(av).localeCompare(String(bv)) * factor;
    });
  }, [rows, columns, sortKey, dir]);

  function toggleSort(col: Column<T>) {
    if (!col.sortable || !col.sortValue) return;
    if (sortKey === col.key) {
      setDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(col.key);
      setDir("desc");
    }
  }

  function onRowKey(e: KeyboardEvent<HTMLTableRowElement>, row: T) {
    if (!onRowClick) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onRowClick(row);
    }
  }

  return (
    <div
      className="overflow-auto overscroll-x-contain"
      style={maxHeight ? { maxHeight } : undefined}
    >
      <table className="w-full border-collapse text-body">
        <thead className="sticky top-0 z-10">
          <tr className="bg-surface-2">
            {columns.map((col, ci) => {
              const active = sortKey === col.key;
              const sticky = stickyFirst && ci === 0;
              const sortable = !!(col.sortable && col.sortValue);
              const align = col.align ?? "left";
              return (
                <th
                  key={col.key}
                  scope="col"
                  aria-sort={
                    active ? (dir === "asc" ? "ascending" : "descending") : undefined
                  }
                  style={col.width ? { width: col.width } : undefined}
                  className={`eyebrow h-9 border-b border-line-strong px-4 whitespace-nowrap pointer-coarse:h-11 ${ALIGN[align]} ${
                    active ? "text-fg" : ""
                  } ${sticky ? STICKY_HEAD : ""} ${col.headerClassName ?? ""}`}
                >
                  {sortable ? (
                    <button
                      type="button"
                      onClick={() => toggleSort(col)}
                      className={`inline-flex w-full items-center gap-1 uppercase transition-colors hover:text-fg pointer-coarse:min-h-11 pointer-coarse:min-w-11 ${JUSTIFY[align]}`}
                    >
                      <span>{col.header}</span>
                      <span aria-hidden className={active ? "text-accent" : "text-transparent"}>
                        {dir === "asc" && active ? "▲" : "▼"}
                      </span>
                    </button>
                  ) : (
                    col.header
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => {
            const active = isActiveRow?.(row) ?? false;
            const clickable = !!onRowClick;
            return (
              <tr
                key={rowKey(row, i)}
                onClick={clickable ? () => onRowClick(row) : undefined}
                onKeyDown={clickable ? (e) => onRowKey(e, row) : undefined}
                tabIndex={clickable ? 0 : undefined}
                aria-label={clickable ? rowActionLabel?.(row) : undefined}
                className={`group border-b border-line/70 transition-colors last:border-b-0 ${
                  clickable ? "cursor-pointer focus-visible:bg-surface-3 focus-visible:outline-none" : ""
                } ${active ? "bg-surface-3" : "hover:bg-surface-2"}`}
              >
                {columns.map((col, ci) => (
                  <td
                    key={col.key}
                    className={`h-10 px-4 whitespace-nowrap pointer-coarse:h-12 ${ALIGN[col.align ?? "left"]} ${
                      stickyFirst && ci === 0 ? (active ? STICKY_CELL_ACTIVE : STICKY_CELL) : ""
                    } ${col.cellClassName ?? ""}`}
                  >
                    {col.render(row)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
