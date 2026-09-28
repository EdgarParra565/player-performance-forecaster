import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Chip, NumberField, Segmented } from "../controls";
import { DataTable, type Column } from "../DataTable";
import { SideTag } from "../Delta";

describe("controls", () => {
  it("NumberField lets you type through intermediate values, clamps on blur", () => {
    const onChange = vi.fn();
    render(<NumberField label="Min books" value={2} onChange={onChange} min={2} max={12} />);
    const input = screen.getByLabelText("Min books");
    fireEvent.change(input, { target: { value: "1" } }); // on the way to "12"
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.change(input, { target: { value: "12" } });
    fireEvent.blur(input);
    expect(onChange).toHaveBeenCalledWith(12);
    fireEvent.change(input, { target: { value: "99" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onChange).toHaveBeenLastCalledWith(12);
  });

  it("NumberField reverts garbage input", () => {
    const onChange = vi.fn();
    render(<NumberField label="Games" value={25} onChange={onChange} />);
    const input = screen.getByLabelText("Games") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "" } });
    fireEvent.blur(input);
    expect(onChange).not.toHaveBeenCalled();
    expect(input.value).toBe("25");
  });

  it("Chip and Segmented expose pressed state", () => {
    render(
      <>
        <Chip active onClick={() => {}}>
          fanduel
        </Chip>
        <Segmented
          label="Mode"
          options={[
            { key: "a", label: "A" },
            { key: "b", label: "B" },
          ]}
          value="b"
          onChange={() => {}}
        />
      </>,
    );
    expect(screen.getByRole("button", { name: "fanduel" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "A" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: "B" })).toHaveAttribute("aria-pressed", "true");
  });

  it("SideTag is neutral text, not a polarity color", () => {
    const { container } = render(<SideTag side="under" />);
    expect(container.innerHTML).not.toMatch(/text-(pos|neg)/);
  });
});

describe("DataTable accessibility", () => {
  const cols: Column<{ n: string; v: number }>[] = [
    { key: "n", header: "Name", render: (r) => r.n },
    { key: "v", header: "Val", render: (r) => String(r.v), sortable: true, sortValue: (r) => r.v },
  ];

  it("clickable rows are keyboard-activatable", () => {
    const onRowClick = vi.fn();
    render(
      <DataTable
        columns={cols}
        rows={[{ n: "A", v: 1 }]}
        rowKey={(r) => r.n}
        onRowClick={onRowClick}
        rowActionLabel={(r) => `Open ${r.n}`}
      />,
    );
    const row = screen.getByRole("row", { name: "Open A" });
    expect(row).toHaveAttribute("tabindex", "0");
    fireEvent.keyDown(row, { key: "Enter" });
    fireEvent.keyDown(row, { key: " " });
    expect(onRowClick).toHaveBeenCalledTimes(2);
  });

  it("sortable headers are buttons and report aria-sort", () => {
    render(<DataTable columns={cols} rows={[{ n: "A", v: 1 }]} rowKey={(r) => r.n} />);
    fireEvent.click(screen.getByRole("button", { name: /Val/ }));
    expect(screen.getByRole("columnheader", { name: /Val/ })).toHaveAttribute("aria-sort", "descending");
  });

  it("two null sort values compare equal (stable order)", () => {
    const rows = [
      { n: "X", v: null as unknown as number },
      { n: "Y", v: null as unknown as number },
      { n: "Z", v: 1 },
    ];
    render(<DataTable columns={cols} rows={rows} rowKey={(r) => r.n} initialSort={{ key: "v", dir: "desc" }} />);
    const texts = screen.getAllByRole("row").slice(1).map((r) => r.textContent);
    expect(texts[0]).toContain("Z");
    expect(texts[1]).toContain("X");
    expect(texts[2]).toContain("Y");
  });
});
