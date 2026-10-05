import { ApiError } from "../api/client";

// App-wide notice for deployment-level database problems. Deliberately NOT
// the offseason empty state: "no lines posted yet" is normal, "no database
// file" means the deploy is broken and nothing in the app is real data.
export function DbNotice({ error }: { error: unknown }) {
  if (!(error instanceof ApiError) || error.status !== 503) return null;
  const notMounted = error.code === "db_not_mounted";
  if (!notMounted && error.code !== "db_invalid") return null;
  return (
    <div role="alert" className="mb-6 rounded-lg border border-neg-dim/60 bg-neg-soft px-4 py-4 text-body">
      <div className="font-semibold text-neg">
        {notMounted ? "Database file not mounted" : "Database unavailable"}
      </div>
      {notMounted ? (
        <div className="mt-2 space-y-2 text-muted">
          <p>
            The API is running but found no SQLite file, so every view is empty. This is a deployment problem, not
            an offseason slate.
          </p>
          <p>Start the container with the database bind-mounted:</p>
          <pre className="tnum overflow-x-auto rounded-md border border-line bg-surface-2 px-3 py-2 text-caption text-fg">
            {`docker compose up flagship
# or
docker run -p 8080:8080 -v "$(pwd)/data/database:/data:ro" \\
  -e NBA_DB_PATH=/data/nba_data.db nba-flagship`}
          </pre>
          <p className="text-faint">On Fly.io: publish a snapshot first (DEPLOYMENT.md §16).</p>
        </div>
      ) : (
        <p className="mt-2 text-muted">
          A file exists at NBA_DB_PATH but it isn't a usable NBA database (corrupt, or missing core tables). Nothing
          was written to it.
        </p>
      )}
    </div>
  );
}
