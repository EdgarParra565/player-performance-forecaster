"""Suite-wide guard: tests must never open the LIVE database.

Several code paths default ``db_path`` to data/database/nba_data.db. Tests that
forgot to pass a temp path silently wrote into the live DB (a fabricated
LeBron game log from a module-level script, and every odds_poll_runs row). This
guard makes any such connection fail loudly instead.
"""

import os
import sqlite3
from pathlib import Path

_LIVE_DB = os.path.realpath(
    Path(__file__).resolve().parents[2] / "data" / "database" / "nba_data.db"
)
_real_connect = sqlite3.connect


def _guarded_connect(database, *args, **kwargs):
    target = str(database)
    if not target.startswith("file:") and target != ":memory:":
        if os.path.realpath(target) == _LIVE_DB:
            raise RuntimeError(
                "test tried to open the LIVE database "
                f"{_LIVE_DB}; pass a temp db_path instead"
            )
    return _real_connect(database, *args, **kwargs)


sqlite3.connect = _guarded_connect
