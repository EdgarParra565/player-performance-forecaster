import { ApiError } from "../api/client";
import { TableSkeleton } from "./Skeleton";

// Back-compat loading block: a table-shaped skeleton.
export function Loading({ rows = 5 }: { rows?: number }) {
  return <TableSkeleton rows={rows} />;
}

// Turn a query error into a human message, keeping the API's `detail`
// (e.g. "model_mode must be one of …", "database not found").
export function errorMessage(error: unknown, fallback: string): string {
  if (error instanceof ApiError && error.message) return `${fallback} ${error.message}`;
  return fallback;
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="flex items-center gap-2 rounded-lg border border-neg-dim/50 bg-neg-soft px-4 py-3 text-body text-neg"
    >
      <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-neg" aria-hidden />
      {message}
    </div>
  );
}
