import { useEffect, useState, type FormEvent } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  ACCESS_REQUIRED_EVENT,
  clearAccessCode,
  getAccessCode,
  setAccessCode,
} from "../lib/access";

// Private-deploy gate: shown when the server has FLAGSHIP_ACCESS_CODE set and
// we have no (valid) code yet. It validates the code against a cheap endpoint
// before storing it, then refetches everything. Not real auth — a shared
// secret for small private deploys (DEPLOYMENT.md §15).
export function AccessGate({ required }: { required: boolean }) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(() => required && !getAccessCode());
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (required && !getAccessCode()) setOpen(true);
  }, [required]);

  useEffect(() => {
    function onRequired() {
      clearAccessCode(); // the stored code was rejected (rotated / wrong)
      setOpen(true);
    }
    window.addEventListener(ACCESS_REQUIRED_EVENT, onRequired);
    return () => window.removeEventListener(ACCESS_REQUIRED_EVENT, onRequired);
  }, []);

  if (!open) return null;

  async function submit(e: FormEvent) {
    e.preventDefault();
    const trimmed = code.trim();
    if (!trimmed) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/meta", {
        headers: { Accept: "application/json", "X-Access-Code": trimmed },
      });
      if (res.ok) {
        setAccessCode(trimmed);
        setOpen(false);
        setCode("");
        await queryClient.invalidateQueries();
      } else if (res.status === 429) {
        setError("Too many attempts. Wait a minute and try again.");
      } else if (res.status === 401) {
        setError("That code isn't right.");
      } else {
        setError(`Server error (${res.status}). Try again shortly.`);
      }
    } catch {
      setError("Couldn't reach the server.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-base/80 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="access-gate-title"
    >
      <form onSubmit={submit} className="panel w-full max-w-sm p-6 shadow-[var(--shadow-pop)]">
        <h2 id="access-gate-title" className="text-title font-semibold text-fg">
          Private preview
        </h2>
        <p className="mt-2 text-body text-muted">
          This deployment is invite-only. Enter the access code you were given.
        </p>
        <label className="mt-4 block text-label text-muted">
          Access code
          <input
            autoFocus
            type="password"
            autoComplete="current-password"
            value={code}
            onChange={(e) => setCode(e.target.value)}
            className="mt-2 h-10 w-full rounded-lg border border-line bg-surface-2 px-3 text-body text-fg focus:border-accent/70 focus:outline-none"
          />
        </label>
        {error && (
          <p role="alert" className="mt-3 text-label text-neg">
            {error}
          </p>
        )}
        <button
          type="submit"
          disabled={busy || !code.trim()}
          className="mt-4 inline-flex h-9 w-full items-center justify-center rounded-lg bg-accent px-4 text-body font-semibold text-base transition-opacity disabled:opacity-40"
        >
          {busy ? "Checking…" : "Continue"}
        </button>
      </form>
    </div>
  );
}
