import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ProbabilityBar } from "../ProbabilityBar";

describe("ProbabilityBar", () => {
  it("renders the percentage label", () => {
    render(<ProbabilityBar value={0.66} />);
    expect(screen.getByText("66.0%")).toBeInTheDocument();
  });

  it("renders an em dash for null values", () => {
    render(<ProbabilityBar value={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("renders a break-even marker tick when a marker is given", () => {
    const { container } = render(
      <ProbabilityBar value={0.6} marker={0.524} />,
    );
    const tick = container.querySelector('[title^="break-even"]');
    expect(tick).not.toBeNull();
  });

  it("omits the numeric label when label=false", () => {
    render(<ProbabilityBar value={0.5} label={false} />);
    expect(screen.queryByText("50.0%")).toBeNull();
  });
});
