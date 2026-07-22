import { useMemo, useState, type ReactNode } from "react";

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
  isActiveRow?: (row: T) => boolean;
  maxHeight?: string;
}

type Dir = "asc" | "desc";

const ALIGN: Record<string, string> = {
  left: "text-left",
  right: "text-right",
  center: "text-center",
};

// Dense, information-first table with a sticky header and click-to-sort. Zebra
// striping is intentionally omitted; hairline row borders + hover carry the
// scanning burden without visual noise.
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  initialSort,
  onRowClick,
  isActiveRow,
  maxHeight,
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(
    initialSort?.key ?? null,
  );
  const [dir, setDir] = useState<Dir>(initialSort?.dir ?? "desc");

  const sorted = useMemo(() => {
    if (!sortKey) return rows;
    const col = columns.find((c) => c.key === sortKey);
    if (!col?.sortValue) return rows;
    const factor = dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const av = col.sortValue!(a);
      const bv = col.sortValue!(b);
      // Nulls always sink to the bottom regardless of direction.
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
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

  return (
    <div
      className="overflow-auto"
      style={maxHeight ? { maxHeight } : undefined}
    >
      <table className="w-full border-collapse text-xs">
        <thead className="sticky top-0 z-10">
          <tr className="bg-panel-2">
            {columns.map((col) => {
              const active = sortKey === col.key;
              return (
                <th
                  key={col.key}
                  onClick={() => toggleSort(col)}
                  style={col.width ? { width: col.width } : undefined}
                  className={`eyebrow border-b border-line-strong px-3 py-2 whitespace-nowrap ${
                    ALIGN[col.align ?? "left"]
                  } ${
                    col.sortable && col.sortValue
                      ? "cursor-pointer select-none hover:text-muted"
                      : ""
                  } ${active ? "text-fg" : ""} ${col.headerClassName ?? ""}`}
                >
                  <span className="inline-flex items-center gap-1">
                    {col.header}
                    {active && (
                      <span className="text-pos">
                        {dir === "asc" ? "▲" : "▼"}
                      </span>
                    )}
                  </span>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => {
            const active = isActiveRow?.(row) ?? false;
            return (
              <tr
                key={rowKey(row, i)}
                onClick={() => onRowClick?.(row)}
                className={`border-b border-line/60 transition-colors ${
                  onRowClick ? "cursor-pointer" : ""
                } ${active ? "bg-panel-3" : "hover:bg-panel-2/70"}`}
              >
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={`px-3 py-1.5 whitespace-nowrap ${
                      ALIGN[col.align ?? "left"]
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
