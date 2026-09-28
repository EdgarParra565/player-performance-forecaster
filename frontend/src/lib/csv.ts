// Client-side CSV export. Mirrors the Streamlit view's "download the scored
// frame" affordance without adding a write path to the API.

export interface CsvColumn<T> {
  key: string;
  header: string;
  value: (row: T) => string | number | null | undefined;
}

// Quote cells containing separators / quotes / newlines, and neutralise
// spreadsheet formula injection: player and book names come from scraped
// pages, so a cell like `=HYPERLINK(...)` must not execute in Excel/Sheets.
export function escapeCell(v: string | number | null | undefined): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : "";
  let s = String(v);
  if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCsv<T>(rows: T[], columns: CsvColumn<T>[]): string {
  const header = columns.map((c) => escapeCell(c.header)).join(",");
  const body = rows.map((r) => columns.map((c) => escapeCell(c.value(r))).join(",")).join("\n");
  return `${header}\n${body}\n`;
}

export function downloadCsv<T>(
  filename: string,
  rows: T[],
  columns: CsvColumn<T>[],
): void {
  const csv = toCsv(rows, columns);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Revoking synchronously can abort the download in some browsers.
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
