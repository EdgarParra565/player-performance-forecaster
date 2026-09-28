import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AccessGate } from "../AccessGate";
import { apiGet } from "../../api/client";
import { getAccessCode, setAccessCode, signalAccessRequired } from "../../lib/access";

function renderGate(required: boolean) {
  const qc = new QueryClient();
  return render(
    <QueryClientProvider client={qc}>
      <AccessGate required={required} />
    </QueryClientProvider>,
  );
}

describe("AccessGate", () => {
  beforeEach(() => window.localStorage.clear());
  afterEach(() => vi.restoreAllMocks());

  it("stays hidden when the server needs no code", () => {
    renderGate(false);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("prompts, validates against the API, then stores the code", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("{}", { status: 401 }))
      .mockResolvedValueOnce(new Response("{}", { status: 200 }));
    renderGate(true);
    const input = screen.getByLabelText("Access code");
    fireEvent.change(input, { target: { value: "wrong" } });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("isn't right");
    expect(getAccessCode()).toBeNull();

    fireEvent.change(input, { target: { value: " right " } });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(getAccessCode()).toBe("right");
    const headers = fetchMock.mock.calls[1][1]?.headers as Record<string, string>;
    expect(headers["X-Access-Code"]).toBe("right");
  });

  it("re-opens and clears a rejected stored code on a 401 signal", () => {
    setAccessCode("stale");
    renderGate(true);
    expect(screen.queryByRole("dialog")).toBeNull();
    act(() => signalAccessRequired());
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(getAccessCode()).toBeNull();
  });
});

describe("api client", () => {
  beforeEach(() => window.localStorage.clear());
  afterEach(() => vi.restoreAllMocks());

  it("sends the stored access code and signals on 401", async () => {
    setAccessCode("abc");
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "access code required" }), { status: 401 }),
    );
    const onRequired = vi.fn();
    window.addEventListener("flagship:access-required", onRequired);
    await expect(apiGet("/meta")).rejects.toThrow("access code required");
    expect((fetchMock.mock.calls[0][1]?.headers as Record<string, string>)["X-Access-Code"]).toBe("abc");
    expect(onRequired).toHaveBeenCalledTimes(1);
    window.removeEventListener("flagship:access-required", onRequired);
  });
});
