import "@testing-library/jest-dom/vitest";

// Newer Node versions expose an experimental global `localStorage` that
// shadows jsdom's and lacks the Storage API unless --localstorage-file is
// set. Install a spec-shaped in-memory Storage for tests.
if (typeof window !== "undefined" && typeof window.localStorage?.clear !== "function") {
  const store = new Map<string, string>();
  const storage: Storage = {
    get length() {
      return store.size;
    },
    clear: () => store.clear(),
    getItem: (k) => (store.has(k) ? store.get(k)! : null),
    key: (i) => Array.from(store.keys())[i] ?? null,
    removeItem: (k) => void store.delete(k),
    setItem: (k, v) => void store.set(k, String(v)),
  };
  Object.defineProperty(window, "localStorage", { value: storage, configurable: true });
  Object.defineProperty(globalThis, "localStorage", { value: storage, configurable: true });
}
