import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DataTable, type Column } from "../DataTable";

interface Row {
  name: string;
  val: number | null;
}

const rows: Row[] = [
  { name: "B", val: 2 },
  { name: "A", val: 1 },
  { name: "C", val: null },
];

const columns: Column<Row>[] = [
  {
    key: "name",
    header: "Name",
    render: (r) => r.name,
    sortable: true,
    sortValue: (r) => r.name,
  },
  {
    key: "val",
    header: "Val",
    align: "right",
    render: (r) => (r.val === null ? "—" : String(r.val)),
    sortable: true,
    sortValue: (r) => r.val,
  },
];

function dataRowTexts(): string[] {
  // Skip the header row.
  return screen
    .getAllByRole("row")
    .slice(1)
    .map((r) => r.textContent ?? "");
}

describe("DataTable", () => {
  it("renders rows in source order when unsorted", () => {
    render(<DataTable columns={columns} rows={rows} rowKey={(r) => r.name} />);
    const texts = dataRowTexts();
    expect(texts[0]).toContain("B");
    expect(texts[1]).toContain("A");
    expect(texts[2]).toContain("C");
  });

  it("sorts descending on first header click, nulls sink", () => {
    render(<DataTable columns={columns} rows={rows} rowKey={(r) => r.name} />);
    fireEvent.click(screen.getByText("Val"));
    const texts = dataRowTexts();
    expect(texts[0]).toContain("2"); // B
    expect(texts[1]).toContain("1"); // A
    expect(texts[2]).toContain("C"); // null sinks to bottom
  });

  it("toggles to ascending on the second click", () => {
    render(<DataTable columns={columns} rows={rows} rowKey={(r) => r.name} />);
    fireEvent.click(screen.getByText("Val"));
    fireEvent.click(screen.getByText("Val"));
    const texts = dataRowTexts();
    expect(texts[0]).toContain("1"); // A
    expect(texts[1]).toContain("2"); // B
    expect(texts[2]).toContain("C"); // null still sinks
  });

  it("fires onRowClick with the clicked row", () => {
    const onRowClick = vi.fn();
    render(
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(r) => r.name}
        onRowClick={onRowClick}
      />,
    );
    fireEvent.click(screen.getByText("B"));
    expect(onRowClick).toHaveBeenCalledWith({ name: "B", val: 2 });
  });

  it("honors initialSort", () => {
    render(
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(r) => r.name}
        initialSort={{ key: "name", dir: "asc" }}
      />,
    );
    const texts = dataRowTexts();
    expect(texts[0]).toContain("A");
    expect(texts[1]).toContain("B");
    expect(texts[2]).toContain("C");
  });
});
