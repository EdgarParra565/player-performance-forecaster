// Thin fetch wrapper. All requests go to `/api` (Vite dev proxy locally,
// same origin in production).
import { getAccessCode, signalAccessRequired } from "../lib/access";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

function buildQuery(params?: Record<string, unknown>): string {
  if (!params) return "";
  const usp = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value)) {
      for (const v of value) usp.append(key, String(v));
    } else {
      usp.append(key, String(value));
    }
  }
  const q = usp.toString();
  return q ? `?${q}` : "";
}

export async function apiGet<T>(
  path: string,
  params?: Record<string, unknown>,
): Promise<T> {
  return handle<T>(await fetch(`/api${path}${buildQuery(params)}`, { headers: headers() }));
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  return handle<T>(
    await fetch(`/api${path}`, {
      method: "POST",
      headers: { ...headers(), "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

function headers(): Record<string, string> {
  const h: Record<string, string> = { Accept: "application/json" };
  const code = getAccessCode();
  if (code) h["X-Access-Code"] = code;
  return h;
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* non-JSON error body */
    }
    if (res.status === 401) signalAccessRequired();
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}
