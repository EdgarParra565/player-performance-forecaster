// Optional shared access code (FLAGSHIP_ACCESS_CODE on the server). Stored
// once in localStorage and sent as X-Access-Code on every API call. When the
// server has no code configured this is inert.

const KEY = "flagship.accessCode";
export const ACCESS_REQUIRED_EVENT = "flagship:access-required";

export function getAccessCode(): string | null {
  try {
    return window.localStorage.getItem(KEY);
  } catch {
    return null; // storage disabled (private mode) -> prompt every session
  }
}

export function setAccessCode(code: string): void {
  try {
    window.localStorage.setItem(KEY, code);
  } catch {
    /* storage unavailable */
  }
}

export function clearAccessCode(): void {
  try {
    window.localStorage.removeItem(KEY);
  } catch {
    /* storage unavailable */
  }
}

export function signalAccessRequired(): void {
  window.dispatchEvent(new Event(ACCESS_REQUIRED_EVENT));
}
