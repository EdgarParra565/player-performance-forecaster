"""Desktop UI for single-prop and parlay NBA props model."""
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, scrolledtext, ttk

from nba_api.stats.static import players as static_players

# Allow direct execution via ".../nba_model/simple_ui.py"
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from nba_model.data.database.db_manager import DatabaseManager  # noqa: E402
from nba_model.run_model import (  # noqa: E402
    DEFAULT_AMERICAN_ODDS,
    DEFAULT_BLOWOUT_PENALTY,
    DEFAULT_BLOWOUT_THRESHOLD,
    DEFAULT_DEFENSE_SENSITIVITY,
    DEFAULT_LEAGUE_AVG_DEF_RATING,
    DEFAULT_OPP_DEF_RATING,
    DEFAULT_PARLAY_LINES,
    DEFAULT_PARLAY_STATS,
    DEFAULT_PLAYER_NAME,
    DEFAULT_POINTS_LINE,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_SINGLE_PROP_DISTRIBUTION,
    DEFAULT_VEGAS_SPREAD,
    SINGLE_PROP_DISTRIBUTION_CHOICES,
    _normalize_parlay_stats,
    _resolve_parlay_american_odds,
    run_parlay_demo,
    run_single_prop,
)

_MANUAL_STAT_ALIASES = {
    "points": "points",
    "point": "points",
    "pts": "points",
    "playerpoints": "points",
    "assists": "assists",
    "assist": "assists",
    "ast": "assists",
    "playerassists": "assists",
    "rebounds": "rebounds",
    "rebound": "rebounds",
    "reb": "rebounds",
    "playerrebounds": "rebounds",
    "pra": "pra",
    "pointsreboundsassists": "pra",
    "playerpointsreboundsassists": "pra",
}

# Sportsbook presets: dropdown list; "Custom" allows typing a different name in the combobox.
SPORTSBOOK_PRESETS = [
    "PrizePicks",
    "Underdog",
    "DraftKings",
    "FanDuel",
    "BetMGM",
    "Fliff",
    "Betr",
    "Fanatics",
    "Caesars",
    "BetRivers",
    "Custom",
]

# Parlay leg presets: label -> (stats list, lines list). User can still edit after selecting.
PARLAY_LEG_PRESETS = {
    "Custom (edit below)": (None, None),
    "PRA (points, rebounds, assists)": (["points", "rebounds", "assists"], [25.5, 8.5, 5.5]),
    "Points + Assists": (["points", "assists"], [25.5, 5.5]),
    "Points + Rebounds": (["points", "rebounds"], [25.5, 8.5]),
    "Assists + Rebounds": (["assists", "rebounds"], [5.5, 8.5]),
    "2-leg PRA (pts, reb)": (["points", "rebounds"], [25.5, 8.5]),
    "2-leg PRA (pts, ast)": (["points", "assists"], [25.5, 5.5]),
    "2-leg PRA (reb, ast)": (["rebounds", "assists"], [8.5, 5.5]),
}

# Single-prop stat type -> default line when user picks that stat.
SINGLE_STAT_DEFAULT_LINES = {
    "points": 25.5,
    "assists": 5.5,
    "rebounds": 8.5,
    "pra": 35.5,
}


class _OperationRunner:
    """Manage one subprocess at a time with live streaming into a Tk widget.

    Output is pushed onto a queue from the reader thread and drained on the Tk
    main loop via root.after, which avoids cross-thread Tk calls.
    """

    def __init__(self, root: tk.Tk, output_widget: scrolledtext.ScrolledText):
        self._root = root
        self._output = output_widget
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._queue: Queue[str] = Queue()
        self._on_done = None
        self._poll_scheduled = False

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, label: str, cmd_args: list[str], cwd: str | None = None,
              env_extra: dict | None = None, on_done=None) -> bool:
        if self.is_running():
            messagebox.showwarning(
                "Operation in progress",
                "Another operation is already running. Stop it first.",
            )
            return False
        self._on_done = on_done
        self._append(f"\n[{datetime.now().strftime('%H:%M:%S')}] $ {label}\n")
        self._append(f"  cmd: {' '.join(shlex.quote(a) for a in cmd_args)}\n")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if env_extra:
            env.update(env_extra)
        try:
            self._proc = subprocess.Popen(
                cmd_args,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                start_new_session=True,
            )
        except Exception as exc:
            self._append(f"  ERROR: failed to launch: {exc}\n")
            return False
        self._reader = threading.Thread(
            target=self._read_loop, args=(self._proc,), daemon=True,
        )
        self._reader.start()
        self._schedule_poll()
        return True

    def stop(self) -> None:
        if not self.is_running():
            self._append("  (no running process to stop)\n")
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            self._append("  >> SIGTERM sent\n")
        except Exception as exc:
            self._append(f"  >> stop failed: {exc}\n")

    def clear(self) -> None:
        self._output.configure(state="normal")
        self._output.delete("1.0", tk.END)
        self._output.configure(state="disabled")

    def _read_loop(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._queue.put(line)
        finally:
            proc.wait()
            self._queue.put(f"\n[exit {proc.returncode}]\n")

    def _schedule_poll(self) -> None:
        if self._poll_scheduled:
            return
        self._poll_scheduled = True
        self._root.after(100, self._drain)

    def _drain(self) -> None:
        self._poll_scheduled = False
        drained_any = False
        try:
            while True:
                line = self._queue.get_nowait()
                self._append(line)
                drained_any = True
        except Empty:
            pass
        running = self.is_running()
        if running or drained_any:
            self._schedule_poll()
        if not running and not drained_any and self._proc is not None:
            rc = self._proc.returncode
            self._proc = None
            if self._on_done:
                cb, self._on_done = self._on_done, None
                try:
                    cb(rc)
                except Exception:
                    pass

    def _append(self, text: str) -> None:
        self._output.configure(state="normal")
        self._output.insert(tk.END, text)
        self._output.see(tk.END)
        self._output.configure(state="disabled")


class SimpleModelUI:
    """Small desktop UI to run single-prop and parlay workflows."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NBA Props - Simple UI")
        self.root.geometry("980x720")
        self.manual_records = []
        self._player_lookup_cache = {}

        self._build_layout()

    def _build_layout(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        single_tab = ttk.Frame(notebook)
        parlay_tab = ttk.Frame(notebook)
        manual_tab = ttk.Frame(notebook)
        charts_tab = ttk.Frame(notebook)
        ops_tab = ttk.Frame(notebook)
        notebook.add(single_tab, text="Single Prop")
        notebook.add(parlay_tab, text="Parlay")
        notebook.add(manual_tab, text="Manual Lines Import")
        notebook.add(charts_tab, text="Player Charts")
        notebook.add(ops_tab, text="Operations")

        self._build_single_tab(single_tab)
        self._build_parlay_tab(parlay_tab)
        self._build_manual_tab(manual_tab)
        self._build_player_charts_tab(charts_tab)
        self._build_operations_tab(ops_tab)

    @staticmethod
    def _add_labeled_entry(parent, row, label, default):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=6, pady=6)
        var = tk.StringVar(value=str(default))
        entry = ttk.Entry(parent, textvariable=var, width=28)
        entry.grid(row=row, column=1, sticky="w", padx=6, pady=6)
        return var

    @staticmethod
    def _add_labeled_slider(
        parent,
        row,
        label,
        default: float,
        min_value: float,
        max_value: float,
        resolution: float = 0.01,
        length: int = 280,
    ):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=6, pady=6)
        container = ttk.Frame(parent)
        container.grid(row=row, column=1, sticky="w", padx=6, pady=6)

        var = tk.DoubleVar(value=float(default))
        decimals = 0
        resolution_text = f"{resolution:.8f}".rstrip("0")
        if "." in resolution_text:
            decimals = len(resolution_text.split(".")[1])
        value_text = tk.StringVar(value=f"{float(default):.{decimals}f}")

        def _on_change(raw_value):
            value_text.set(f"{float(raw_value):.{decimals}f}")

        def _sync_from_var(*_):
            _on_change(var.get())

        slider = tk.Scale(
            container,
            from_=float(min_value),
            to=float(max_value),
            orient=tk.HORIZONTAL,
            resolution=float(resolution),
            variable=var,
            command=_on_change,
            showvalue=False,
            length=length,
        )
        slider.pack(side="left")
        ttk.Label(container, textvariable=value_text, width=7,
                  anchor="e").pack(side="left", padx=(8, 0))
        var.trace_add("write", _sync_from_var)
        _sync_from_var()
        return var

    @staticmethod
    def _parse_csv_strings(value: str):
        return [v.strip() for v in value.split(",") if v.strip()]

    @staticmethod
    def _parse_csv_floats(value: str):
        values = [v.strip() for v in value.split(",") if v.strip()]
        return [float(v) for v in values]

    @staticmethod
    def _parse_float_or_default(value: str, default: float) -> float:
        text = str(value).strip()
        if text == "":
            return float(default)
        return float(text)

    @staticmethod
    def _parse_int_or_default(value: str, default: int) -> int:
        text = str(value).strip()
        if text == "":
            return int(default)
        return int(float(text))

    @staticmethod
    def _canonical_stat_key(value: str) -> str:
        return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())

    @staticmethod
    def _slug_stat_type(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_",
                      str(value).strip().lower()).strip("_")
        return slug or "custom_stat"

    def _normalize_stat_type(self, value: str, allow_custom: bool = False) -> str:
        key = self._canonical_stat_key(value)
        stat_type = _MANUAL_STAT_ALIASES.get(key)
        if not stat_type:
            if allow_custom:
                return self._slug_stat_type(value)
            supported = sorted(set(_MANUAL_STAT_ALIASES.values()))
            raise ValueError(
                f"Unsupported stat '{value}'. Supported: {supported}")
        return stat_type

    @staticmethod
    def _parse_optional_american_odds(value: str):
        text = str(value or "").strip()
        if not text or text.lower() in {"none", "null", "na", "n/a", "-"}:
            return None
        if text.lower() == "even":
            return 100
        try:
            odds = int(round(float(text)))
        except ValueError as exc:
            raise ValueError(f"Invalid American odds '{value}'") from exc
        if odds == 0:
            raise ValueError("American odds cannot be 0")
        return odds

    @staticmethod
    def _tokenize_manual_line(raw_line: str) -> list[str]:
        if "|" in raw_line:
            parts = [part.strip() for part in raw_line.split("|")]
        elif "\t" in raw_line:
            parts = [part.strip() for part in raw_line.split("\t")]
        else:
            parts = next(csv.reader([raw_line], skipinitialspace=True))
            parts = [part.strip() for part in parts]
        return [part for part in parts if part != ""]

    @staticmethod
    def _looks_like_header(parts: list[str]) -> bool:
        if len(parts) < 3:
            return False
        lower_parts = [p.strip().lower() for p in parts[:5]]
        return (
            lower_parts[0] in {"player", "player_name", "name"}
            and any("stat" in token for token in lower_parts)
            and any("line" in token for token in lower_parts)
        )

    @staticmethod
    def _is_date_token(value: str) -> bool:
        value = str(value).strip()
        if not value:
            return False
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _normalize_game_date(value: str) -> str:
        token = str(value).strip()
        if not token:
            raise ValueError("Game date is required (YYYY-MM-DD)")
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(token, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(token.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                f"Invalid date '{value}'. Use YYYY-MM-DD format.") from exc

    @staticmethod
    def _synthetic_player_id(player_name: str) -> int:
        """
        Deterministic synthetic player id for non-NBA names.

        Keeps manual import usable for non-NBA boards while avoiding collisions
        with normal NBA IDs by using a high numeric range.
        """
        digest = sha1(player_name.strip().lower().encode(
            "utf-8")).hexdigest()[:10]
        return 900_000_000 + (int(digest, 16) % 99_000_000)

    @staticmethod
    def _is_matchup_line(value: str) -> bool:
        token = str(value or "").strip().lower()
        return token.startswith("vs ") or token.startswith("@ ")

    @staticmethod
    def _is_team_role_line(value: str) -> bool:
        token = str(value or "").strip()
        if " - " not in token:
            return False
        role = token.split(" - ", 1)[1].strip().lower()
        return any(
            key in role
            for key in (
                "attacker",
                "defender",
                "midfielder",
                "goalkeeper",
                "guard",
                "forward",
                "center",
                "wing",
            )
        )

    @staticmethod
    def _is_noise_line(value: str) -> bool:
        token = str(value or "").strip()
        if not token:
            return True
        lower = token.lower()
        if re.match(r"^\d+(\.\d+)?k$", lower):
            return True
        if re.match(r"^\$\d+(\.\d+)?$", lower):
            return True
        noise_tokens = {
            "refresh board",
            "scoring chart",
            "how to play",
            "help center",
            "enable accessibility",
            "board",
            "my lineups",
            "promotions",
            "invite friends",
            "get $25",
            "ep",
            "demongoblin",
            "demon",
            "goblin",
            "trending",
            "swap",
            "less",
            "more",
            "privacy policy",
            "your privacy choices",
            "responsible gaming",
            "prizepicks blog",
            "terms",
            "careers",
            "contact us",
            "accessibility statement",
        }
        return lower in noise_tokens

    def _resolve_player_identity(self, player_name: str, allow_synthetic: bool = False) -> dict:
        raw_name = str(player_name).strip()
        if not raw_name:
            raise ValueError("Player name is required")
        if raw_name in self._player_lookup_cache:
            return self._player_lookup_cache[raw_name]

        candidates = static_players.find_players_by_full_name(raw_name)
        if not candidates:
            candidates = static_players.find_players_by_full_name(
                raw_name.replace(".", ""))
        if not candidates:
            if allow_synthetic:
                resolved = {"player_id": self._synthetic_player_id(
                    raw_name), "player_name": raw_name}
                self._player_lookup_cache[raw_name] = resolved
                return resolved
            raise ValueError(
                f"Player '{raw_name}' not found in NBA player list")

        exact = next((c for c in candidates if c.get(
            "full_name", "").lower() == raw_name.lower()), candidates[0])
        resolved = {"player_id": int(
            exact["id"]), "player_name": exact["full_name"]}
        self._player_lookup_cache[raw_name] = resolved
        return resolved

    def _parse_manual_line_to_record(
        self,
        parts: list[str],
        default_game_date: str,
        default_book: str,
    ) -> dict:
        if len(parts) < 3:
            raise ValueError("Expected at least 3 fields: player, stat, line")

        has_explicit_date = len(parts) >= 5 and self._is_date_token(parts[1])
        if has_explicit_date:
            player_name = parts[0]
            game_date = self._normalize_game_date(parts[1])
            book = (parts[2] or default_book).strip()
            stat_token = parts[3]
            line_token = parts[4]
            over_token = parts[5] if len(parts) >= 6 else None
            under_token = parts[6] if len(parts) >= 7 else None
        else:
            player_name = parts[0]
            game_date = default_game_date
            book = default_book
            stat_token = parts[1]
            line_token = parts[2]
            over_token = parts[3] if len(parts) >= 4 else None
            under_token = parts[4] if len(parts) >= 5 else None

        if not book:
            raise ValueError("Sportsbook is required (row field or default)")
        try:
            line_value = float(line_token)
        except ValueError as exc:
            raise ValueError(f"Invalid line value '{line_token}'") from exc

        player = self._resolve_player_identity(
            player_name, allow_synthetic=True)
        stat_type = self._normalize_stat_type(stat_token, allow_custom=True)
        over_odds = self._parse_optional_american_odds(over_token)
        under_odds = self._parse_optional_american_odds(under_token)

        return {
            "player_id": player["player_id"],
            "player_name": player["player_name"],
            "game_date": game_date,
            "book": book,
            "stat_type": stat_type,
            "line_value": line_value,
            "over_odds": over_odds,
            "under_odds": under_odds,
        }

    def _parse_board_style_text(self, lines: list[str], default_game_date: str, default_book: str):
        """
        Parse noisy sportsbook board text by locating repeating blocks like:
        player -> matchup -> line -> stat -> Less/More.
        """
        records = []
        errors = []
        seen = set()
        number_re = re.compile(r"^\d+(?:\.\d+)?$")

        for idx, token in enumerate(lines):
            line_value_token = token.strip()
            if not number_re.match(line_value_token):
                continue
            if idx + 1 >= len(lines):
                continue

            stat_token = lines[idx + 1].strip()
            if (
                not stat_token
                or self._is_noise_line(stat_token)
                or self._is_matchup_line(stat_token)
            ):
                continue

            matchup_idx = None
            for j in range(idx - 1, max(-1, idx - 8), -1):
                if self._is_matchup_line(lines[j]):
                    matchup_idx = j
                    break
            if matchup_idx is None:
                continue

            player_name = None
            for j in range(matchup_idx - 1, max(-1, matchup_idx - 8), -1):
                candidate = lines[j].strip()
                if (
                    not candidate
                    or self._is_noise_line(candidate)
                    or self._is_matchup_line(candidate)
                    or self._is_team_role_line(candidate)
                    or number_re.match(candidate)
                ):
                    continue
                player_name = re.sub(
                    r"(Goblin|Demon)+$", "", candidate, flags=re.IGNORECASE).strip()
                if player_name:
                    break
            if not player_name:
                continue

            try:
                player = self._resolve_player_identity(
                    player_name, allow_synthetic=True)
                stat_type = self._normalize_stat_type(
                    stat_token, allow_custom=True)
                line_value = float(line_value_token)
            except Exception as exc:
                errors.append(
                    f"board parse near '{player_name}' / '{stat_token}': {exc}")
                continue

            record = {
                "player_id": player["player_id"],
                "player_name": player["player_name"],
                "game_date": default_game_date,
                "book": default_book,
                "stat_type": stat_type,
                "line_value": line_value,
                "over_odds": None,
                "under_odds": None,
            }
            dedupe_key = (
                record["player_id"],
                record["game_date"],
                record["book"],
                record["stat_type"],
                record["line_value"],
                record["over_odds"],
                record["under_odds"],
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            records.append(record)
        return records, errors

    def _parse_manual_lines_text(self, text: str, default_game_date: str, default_book: str):
        records = []
        errors = []
        normalized_date = self._normalize_game_date(default_game_date)
        normalized_book = (default_book or "").strip() or "manual_ui"
        unstructured_lines = []
        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            has_structured_delimiter = ("|" in stripped) or (
                "\t" in stripped) or (stripped.count(",") >= 2)
            if not has_structured_delimiter:
                unstructured_lines.append(stripped)
                continue
            try:
                parts = self._tokenize_manual_line(stripped)
                if not parts or self._looks_like_header(parts):
                    continue
                record = self._parse_manual_line_to_record(
                    parts,
                    default_game_date=normalized_date,
                    default_book=normalized_book,
                )
                records.append(record)
            except Exception as exc:  # surface all parsing issues together
                errors.append(f"line {line_no}: {exc} | raw='{stripped}'")

        if unstructured_lines:
            board_records, board_errors = self._parse_board_style_text(
                unstructured_lines,
                default_game_date=normalized_date,
                default_book=normalized_book,
            )
            existing = {
                (
                    row["player_id"],
                    row["game_date"],
                    row["book"],
                    row["stat_type"],
                    row["line_value"],
                    row.get("over_odds"),
                    row.get("under_odds"),
                )
                for row in records
            }
            for row in board_records:
                key = (
                    row["player_id"],
                    row["game_date"],
                    row["book"],
                    row["stat_type"],
                    row["line_value"],
                    row.get("over_odds"),
                    row.get("under_odds"),
                )
                if key not in existing:
                    records.append(row)
                    existing.add(key)
            errors.extend(board_errors)

        if not records and unstructured_lines and not errors:
            errors.append(
                "No parseable props. Use delimiter format or paste board text "
                "with player + matchup + line + stat."
            )
        return records, errors

    def _render_output(self, widget: scrolledtext.ScrolledText, payload):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, json.dumps(payload, indent=2))
        widget.configure(state="disabled")

    def _refresh_manual_table(self):
        """Refresh manual parsed rows table from in-memory records."""
        if not hasattr(self, "manual_table"):
            return
        self.manual_table.delete(*self.manual_table.get_children())
        for idx, row in enumerate(self.manual_records):
            self.manual_table.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    idx + 1,
                    row.get("player_name"),
                    row.get("game_date"),
                    row.get("book"),
                    row.get("stat_type"),
                    row.get("line_value"),
                    row.get("over_odds"),
                    row.get("under_odds"),
                ),
            )

    def _delete_selected_manual_rows(self):
        """Delete selected rows from parsed manual records table."""
        selected = list(self.manual_table.selection())
        if not selected:
            messagebox.showinfo(
                "Manual Lines", "Select one or more parsed rows to delete.")
            return
        indexes = sorted((int(iid) for iid in selected), reverse=True)
        for idx in indexes:
            if 0 <= idx < len(self.manual_records):
                del self.manual_records[idx]
        self._refresh_manual_table()
        self._render_output(
            self.manual_output,
            {
                "parsed_records_remaining": len(self.manual_records),
                "deleted_rows": len(selected),
                "sample_records": self.manual_records[:15],
            },
        )

    def _build_single_tab(self, tab):
        controls = ttk.Frame(tab)
        controls.pack(fill="x", padx=8, pady=8)
        controls.columnconfigure(1, weight=1)

        self.single_player = self._add_labeled_entry(
            controls, 0, "Player", DEFAULT_PLAYER_NAME)

        ttk.Label(controls, text="Stat type").grid(
            row=1, column=0, sticky="w", padx=6, pady=6)
        self.single_stat_type = tk.StringVar(value="points")

        def _on_single_stat_change(*_):
            st = self.single_stat_type.get().strip().lower()
            if st in SINGLE_STAT_DEFAULT_LINES:
                self.single_line.set(str(SINGLE_STAT_DEFAULT_LINES[st]))

        single_stat_box = ttk.Combobox(
            controls,
            textvariable=self.single_stat_type,
            values=list(SINGLE_STAT_DEFAULT_LINES.keys()),
            state="readonly",
            width=25,
        )
        single_stat_box.grid(row=1, column=1, sticky="w", padx=6, pady=6)
        self.single_stat_type.trace_add("write", _on_single_stat_change)

        self.single_line = self._add_labeled_entry(
            controls, 2, "Line", DEFAULT_POINTS_LINE)
        self.single_odds = self._add_labeled_entry(
            controls, 3, "American Odds", DEFAULT_AMERICAN_ODDS)
        self.single_window = self._add_labeled_entry(
            controls, 4, "Rolling Window", DEFAULT_ROLLING_WINDOW)
        self.single_opp_def = self._add_labeled_entry(
            controls, 5, "Opponent Def Rating", DEFAULT_OPP_DEF_RATING)
        self.single_spread = self._add_labeled_entry(
            controls, 6, "Vegas Spread", DEFAULT_VEGAS_SPREAD)
        self.single_n_games = self._add_labeled_entry(
            controls, 7, "History Games", 100)
        ttk.Label(controls, text="Distribution").grid(
            row=8, column=0, sticky="w", padx=6, pady=6)
        self.single_distribution = tk.StringVar(
            value=DEFAULT_SINGLE_PROP_DISTRIBUTION)
        distribution_box = ttk.Combobox(
            controls,
            textvariable=self.single_distribution,
            values=SINGLE_PROP_DISTRIBUTION_CHOICES,
            state="readonly",
            width=25,
        )
        distribution_box.grid(row=8, column=1, sticky="w", padx=6, pady=6)
        self.single_league_avg_def = self._add_labeled_slider(
            controls,
            9,
            "League Avg Def Rating",
            DEFAULT_LEAGUE_AVG_DEF_RATING,
            min_value=105.0,
            max_value=120.0,
            resolution=0.1,
        )
        self.single_defense_sensitivity = self._add_labeled_slider(
            controls,
            10,
            "Defense Sensitivity",
            DEFAULT_DEFENSE_SENSITIVITY,
            min_value=0.0,
            max_value=1.5,
            resolution=0.01,
        )
        self.single_blowout_threshold = self._add_labeled_slider(
            controls,
            11,
            "Blowout Threshold",
            DEFAULT_BLOWOUT_THRESHOLD,
            min_value=0.0,
            max_value=25.0,
            resolution=0.5,
        )
        self.single_blowout_penalty = self._add_labeled_slider(
            controls,
            12,
            "Blowout Penalty",
            DEFAULT_BLOWOUT_PENALTY,
            min_value=0.0,
            max_value=0.5,
            resolution=0.01,
        )
        self.single_defense_severity = self._add_labeled_slider(
            controls,
            13,
            "Defense Severity",
            1.0,
            min_value=0.0,
            max_value=3.0,
            resolution=0.05,
        )
        self.single_minutes_severity = self._add_labeled_slider(
            controls,
            14,
            "Minutes Penalty Severity",
            1.0,
            min_value=0.0,
            max_value=3.0,
            resolution=0.05,
        )
        self.single_sigma_severity = self._add_labeled_slider(
            controls,
            15,
            "Volatility (Sigma) Severity",
            1.0,
            min_value=0.0,
            max_value=3.0,
            resolution=0.05,
        )

        ttk.Button(
            controls,
            text="Reset Single Tuning Defaults",
            command=self._reset_single_tuning_defaults,
        ).grid(row=16, column=0, sticky="w", padx=6, pady=6)

        self.single_show_plot = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Show Distribution Plot",
            variable=self.single_show_plot,
        ).grid(row=17, column=0, sticky="w", padx=6, pady=6)

        ttk.Button(controls, text="Run Single Prop", command=self._run_single).grid(
            row=17, column=1, sticky="w", padx=6, pady=6
        )

        self.single_output = scrolledtext.ScrolledText(
            tab, height=18, wrap=tk.WORD, state="disabled")
        self.single_output.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _build_parlay_tab(self, tab):
        controls = ttk.Frame(tab)
        controls.pack(fill="x", padx=8, pady=8)
        controls.columnconfigure(1, weight=1)

        self.parlay_player = self._add_labeled_entry(
            controls, 0, "Player", DEFAULT_PLAYER_NAME)

        ttk.Label(controls, text="Sportsbook").grid(
            row=1, column=0, sticky="w", padx=6, pady=6)
        self.parlay_sportsbook = tk.StringVar(value="PrizePicks")
        sportsbook_box = ttk.Combobox(
            controls,
            textvariable=self.parlay_sportsbook,
            values=SPORTSBOOK_PRESETS,
            state="normal",
            width=25,
        )
        sportsbook_box.grid(row=1, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(controls, text="Leg preset").grid(
            row=2, column=0, sticky="w", padx=6, pady=6)
        self.parlay_leg_preset = tk.StringVar(
            value="PRA (points, rebounds, assists)")

        def _on_parlay_preset_change(*_):
            key = self.parlay_leg_preset.get()
            preset = PARLAY_LEG_PRESETS.get(key)
            if preset and preset[0] is not None and preset[1] is not None:
                self.parlay_stats.set(", ".join(preset[0]))
                self.parlay_lines.set(", ".join(str(x) for x in preset[1]))

        parlay_preset_box = ttk.Combobox(
            controls,
            textvariable=self.parlay_leg_preset,
            values=list(PARLAY_LEG_PRESETS.keys()),
            state="readonly",
            width=28,
        )
        parlay_preset_box.grid(row=2, column=1, sticky="w", padx=6, pady=6)
        self.parlay_leg_preset.trace_add("write", _on_parlay_preset_change)

        stats_default, lines_default = PARLAY_LEG_PRESETS["PRA (points, rebounds, assists)"]
        self.parlay_stats = self._add_labeled_entry(
            controls, 3, "Leg Stats (csv)",
            ", ".join(stats_default) if stats_default else ", ".join(
                DEFAULT_PARLAY_STATS),
        )
        self.parlay_lines = self._add_labeled_entry(
            controls, 4, "Leg Lines (csv)",
            ", ".join(str(x) for x in lines_default) if lines_default else ", ".join(
                str(x) for x in DEFAULT_PARLAY_LINES),
        )

        ttk.Label(controls, text="Parlay Odds Format").grid(
            row=5, column=0, sticky="w", padx=6, pady=6)
        self.parlay_odds_format = tk.StringVar(value="multiplier")
        odds_format_box = ttk.Combobox(
            controls,
            textvariable=self.parlay_odds_format,
            values=["american", "decimal", "multiplier"],
            state="readonly",
            width=25,
        )
        odds_format_box.grid(row=5, column=1, sticky="w", padx=6, pady=6)

        self.parlay_odds = self._add_labeled_entry(
            controls, 6, "Parlay Odds", 3.0)
        self.parlay_n_games = self._add_labeled_entry(
            controls, 7, "History Games", 100)
        self.parlay_n_sims = self._add_labeled_entry(
            controls, 8, "Simulation Runs", 20000)
        self.parlay_corr_severity = self._add_labeled_slider(
            controls,
            9,
            "Correlation Severity",
            1.0,
            min_value=0.0,
            max_value=3.0,
            resolution=0.05,
        )
        self.parlay_vol_severity = self._add_labeled_slider(
            controls,
            10,
            "Volatility Severity",
            1.0,
            min_value=0.0,
            max_value=3.0,
            resolution=0.05,
        )

        ttk.Button(
            controls,
            text="Reset Parlay Tuning Defaults",
            command=self._reset_parlay_tuning_defaults,
        ).grid(row=11, column=0, sticky="w", padx=6, pady=6)

        ttk.Button(controls, text="Run Parlay", command=self._run_parlay).grid(
            row=11, column=1, sticky="w", padx=6, pady=6
        )

        self.parlay_output = scrolledtext.ScrolledText(
            tab, height=16, wrap=tk.WORD, state="disabled")
        self.parlay_output.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _build_manual_tab(self, tab):
        controls = ttk.Frame(tab)
        controls.pack(fill="x", padx=8, pady=8)

        self.manual_default_date = self._add_labeled_entry(
            controls, 0, "Default Game Date (YYYY-MM-DD)", datetime.utcnow().strftime(
                "%Y-%m-%d")
        )
        ttk.Label(controls, text="Default Sportsbook").grid(
            row=1, column=0, sticky="w", padx=6, pady=6)
        self.manual_default_book = tk.StringVar(value="Manual import")
        manual_book_box = ttk.Combobox(
            controls,
            textvariable=self.manual_default_book,
            values=SPORTSBOOK_PRESETS + ["Manual import"],
            state="normal",
            width=28,
        )
        manual_book_box.grid(row=1, column=1, sticky="w", padx=6, pady=6)
        self.manual_db_path = self._add_labeled_entry(
            controls, 2, "DB Path", "data/database/nba_data.db")

        ttk.Button(controls, text="Parse Lines", command=self._parse_manual_lines).grid(
            row=3, column=0, sticky="w", padx=6, pady=6
        )
        ttk.Button(
            controls,
            text="Save Parsed Lines to DB",
            command=self._save_manual_lines,
        ).grid(row=3, column=1, sticky="w", padx=6, pady=6)
        ttk.Button(
            controls,
            text="Delete Selected Parsed Row(s)",
            command=self._delete_selected_manual_rows,
        ).grid(row=3, column=2, sticky="w", padx=6, pady=6)

        help_text = (
            "Paste one prop per line (pipe, csv, or tab-delimited).\n"
            "Formats:\n"
            "  1) player | stat | line | over_odds | under_odds\n"
            "  2) player | game_date | book | stat | line | over_odds | under_odds\n"
            "  3) Raw board dump (auto-extract from blocks: player + matchup "
            "+ line + stat)\n"
            "Notes: stats accept aliases (pts/ast/reb/pra). "
            "Unknown stats become snake_case. Odds are optional."
        )
        ttk.Label(tab, text=help_text, justify="left").pack(
            anchor="w", padx=8, pady=(0, 6))

        self.manual_input = scrolledtext.ScrolledText(
            tab, height=12, wrap=tk.WORD)
        self.manual_input.pack(fill="x", expand=False, padx=8, pady=(0, 8))
        self.manual_input.insert(
            tk.END,
            "# Example rows\n"
            "LeBron James | points | 27.5 | -115 | -105\n"
            "Stephen Curry | 2025-03-07 | FanDuel | assists | 5.5 | -110 | -120\n",
        )

        table_frame = ttk.Frame(tab)
        table_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        columns = ("row", "player", "date", "book",
                   "stat", "line", "over", "under")
        self.manual_table = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=10)
        self.manual_table.heading("row", text="#")
        self.manual_table.heading("player", text="Player")
        self.manual_table.heading("date", text="Game Date")
        self.manual_table.heading("book", text="Book")
        self.manual_table.heading("stat", text="Stat")
        self.manual_table.heading("line", text="Line")
        self.manual_table.heading("over", text="Over Odds")
        self.manual_table.heading("under", text="Under Odds")

        self.manual_table.column("row", width=40, anchor="center")
        self.manual_table.column("player", width=170)
        self.manual_table.column("date", width=95, anchor="center")
        self.manual_table.column("book", width=120)
        self.manual_table.column("stat", width=130)
        self.manual_table.column("line", width=80, anchor="e")
        self.manual_table.column("over", width=90, anchor="center")
        self.manual_table.column("under", width=90, anchor="center")

        yscroll = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.manual_table.yview)
        self.manual_table.configure(yscrollcommand=yscroll.set)
        self.manual_table.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        self.manual_output = scrolledtext.ScrolledText(
            tab, height=12, wrap=tk.WORD, state="disabled")
        self.manual_output.pack(fill="both", expand=False, padx=8, pady=(0, 8))

    def _parse_manual_lines(self):
        try:
            text = self.manual_input.get("1.0", tk.END)
            records, errors = self._parse_manual_lines_text(
                text=text,
                default_game_date=self.manual_default_date.get().strip(),
                default_book=self.manual_default_book.get().strip(),
            )
            self.manual_records = records
            self._refresh_manual_table()
            payload = {
                "parsed_records": len(records),
                "errors": errors,
                "sample_records": records[:15],
            }
            self._render_output(self.manual_output, payload)
            if errors:
                messagebox.showwarning(
                    "Manual Lines Parse",
                    "Parsed %s line(s) with %s error(s). Review output for details."
                    % (len(records), len(errors)),
                )
            elif records:
                messagebox.showinfo(
                    "Manual Lines Parse",
                    "Parsed %s line(s). Click 'Save Parsed Lines to DB' to store."
                    % len(records),
                )
            else:
                messagebox.showinfo(
                    "Manual Lines Parse", "No valid lines were found in the input box.")
        except Exception as exc:
            messagebox.showerror("Manual Lines Parse Error", str(exc))

    def _save_manual_lines(self):
        try:
            if not self.manual_records:
                raise ValueError(
                    "No parsed lines available. Click 'Parse Lines' first.")

            db_path = self.manual_db_path.get().strip() or "data/database/nba_data.db"
            with DatabaseManager(db_path=db_path) as db:
                before_count = db.conn.execute(
                    "SELECT COUNT(*) FROM betting_lines").fetchone()[0]
                seen = {}
                for row in self.manual_records:
                    player_id = row["player_id"]
                    if player_id not in seen:
                        seen[player_id] = row["player_name"]
                        db.insert_player(player_id, row["player_name"])
                db.insert_betting_lines_records(self.manual_records)
                after_count = db.conn.execute(
                    "SELECT COUNT(*) FROM betting_lines").fetchone()[0]

            inserted = after_count - before_count
            payload = {
                "db_path": db_path,
                "parsed_rows_submitted": len(self.manual_records),
                "new_rows_inserted": inserted,
                "total_betting_lines_rows": after_count,
            }
            self._render_output(self.manual_output, payload)
            msg = (
                "Saved %s parsed row(s). Inserted %s new betting_lines row(s)."
                % (len(self.manual_records), inserted)
            )
            messagebox.showinfo("Manual Lines Save", msg)
        except Exception as exc:
            messagebox.showerror("Manual Lines Save Error", str(exc))

    # ---- Player Charts tab -------------------------------------------------

    def _build_player_charts_tab(self, tab):
        """Charts: recent stats, distribution, splits, hit-rate, EV per book."""
        # Lazy import so the rest of the UI loads even if matplotlib is missing.
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from nba_model.visualization import player_charts

        self._mpl_canvas_cls = FigureCanvasTkAgg
        self._mpl_figure_cls = Figure
        self._player_charts_mod = player_charts

        outer = ttk.Frame(tab)
        outer.pack(fill="both", expand=True, padx=8, pady=8)

        # --- Selection panel ----------------------------------------------
        controls = ttk.LabelFrame(outer, text="Selection")
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        ttk.Label(controls, text="DB path").grid(
            row=0, column=0, sticky="w", padx=6, pady=4)
        self.charts_db_path = tk.StringVar(value="data/database/nba_data.db")
        ttk.Entry(controls, textvariable=self.charts_db_path, width=42).grid(
            row=0, column=1, columnspan=3, sticky="we", padx=6, pady=4)
        ttk.Button(controls, text="Reload teams + players",
                   command=self._charts_reload_lookups).grid(
            row=0, column=4, padx=6)

        ttk.Label(controls, text="Team").grid(
            row=1, column=0, sticky="w", padx=6, pady=4)
        self.charts_team = tk.StringVar(value="(any)")
        self.charts_team_box = ttk.Combobox(
            controls, textvariable=self.charts_team, state="readonly", width=12,
        )
        self.charts_team_box.grid(row=1, column=1, sticky="w", padx=6, pady=4)
        self.charts_team_box.bind(
            "<<ComboboxSelected>>", lambda *_: self._charts_refresh_player_box())

        ttk.Label(controls, text="Player").grid(
            row=1, column=2, sticky="w", padx=6, pady=4)
        self.charts_player = tk.StringVar(value="")
        self.charts_player_box = ttk.Combobox(
            controls, textvariable=self.charts_player, width=28,
        )
        self.charts_player_box.grid(row=1, column=3, sticky="we", padx=6, pady=4)

        ttk.Label(controls, text="Stat").grid(
            row=2, column=0, sticky="w", padx=6, pady=4)
        self.charts_stat = tk.StringVar(value="points")
        ttk.Combobox(
            controls, textvariable=self.charts_stat, state="readonly",
            values=["points", "assists", "rebounds", "pra", "minutes"],
            width=12,
        ).grid(row=2, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(controls, text="Last N games").grid(
            row=2, column=2, sticky="w", padx=6, pady=4)
        self.charts_n_games = tk.StringVar(value="25")
        ttk.Entry(controls, textvariable=self.charts_n_games, width=6).grid(
            row=2, column=3, sticky="w", padx=6, pady=4)

        ttk.Label(controls, text="Rolling window").grid(
            row=3, column=0, sticky="w", padx=6, pady=4)
        self.charts_rolling = tk.StringVar(value="5")
        ttk.Entry(controls, textvariable=self.charts_rolling, width=6).grid(
            row=3, column=1, sticky="w", padx=6, pady=4)

        # Distribution overlays
        dist_box = ttk.Frame(controls)
        dist_box.grid(row=3, column=2, columnspan=2, sticky="we", padx=6, pady=4)
        ttk.Label(dist_box, text="Overlays:").pack(side="left")
        self.charts_dist_normal = tk.BooleanVar(value=True)
        self.charts_dist_poisson = tk.BooleanVar(value=False)
        self.charts_dist_negbin = tk.BooleanVar(value=False)
        ttk.Checkbutton(dist_box, text="normal",
                        variable=self.charts_dist_normal).pack(side="left", padx=4)
        ttk.Checkbutton(dist_box, text="poisson",
                        variable=self.charts_dist_poisson).pack(side="left", padx=4)
        ttk.Checkbutton(dist_box, text="neg-binomial",
                        variable=self.charts_dist_negbin).pack(side="left", padx=4)

        ttk.Button(controls, text="Show charts",
                   command=self._charts_render).grid(
            row=4, column=3, sticky="e", padx=6, pady=4)

        # --- Sub-notebook with view tabs ----------------------------------
        view_nb = ttk.Notebook(outer)
        view_nb.pack(fill="both", expand=True, pady=(8, 0))

        overview_view = ttk.Frame(view_nb)
        splits_view = ttk.Frame(view_nb)
        hitrate_view = ttk.Frame(view_nb)
        view_nb.add(overview_view, text="Overview")
        view_nb.add(splits_view, text="Splits")
        view_nb.add(hitrate_view, text="Hit Rate + Custom Line")

        # Overview view: recent games + distribution stacked
        overview_view.columnconfigure(0, weight=1)
        overview_view.rowconfigure(0, weight=1)
        overview_view.rowconfigure(1, weight=1)
        self._charts_top_holder = ttk.Frame(overview_view)
        self._charts_top_holder.grid(row=0, column=0, sticky="nsew")
        self._charts_bottom_holder = ttk.Frame(overview_view)
        self._charts_bottom_holder.grid(row=1, column=0, sticky="nsew",
                                        pady=(6, 0))

        # Splits view: single splits figure
        splits_view.columnconfigure(0, weight=1)
        splits_view.rowconfigure(0, weight=1)
        self._charts_splits_holder = ttk.Frame(splits_view)
        self._charts_splits_holder.grid(row=0, column=0, sticky="nsew")

        # Hit rate view: chart + custom-line probe panel
        hitrate_view.columnconfigure(0, weight=1)
        hitrate_view.rowconfigure(0, weight=1)
        self._charts_hitrate_holder = ttk.Frame(hitrate_view)
        self._charts_hitrate_holder.grid(row=0, column=0, sticky="nsew")
        probe = ttk.LabelFrame(hitrate_view, text="Custom line probe")
        probe.grid(row=1, column=0, sticky="we", pady=(6, 0))
        ttk.Label(probe, text="Line").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        self.charts_custom_line = tk.StringVar(value="")
        ttk.Entry(probe, textvariable=self.charts_custom_line, width=8).grid(
            row=0, column=1, padx=6, pady=4)
        ttk.Label(probe, text="Over odds").grid(row=0, column=2, padx=6, pady=4, sticky="w")
        self.charts_custom_odds = tk.StringVar(value="-110")
        ttk.Entry(probe, textvariable=self.charts_custom_odds, width=8).grid(
            row=0, column=3, padx=6, pady=4)
        ttk.Button(probe, text="Evaluate",
                   command=self._charts_evaluate_custom).grid(
            row=0, column=4, padx=6, pady=4)
        self.charts_probe_result = tk.StringVar(
            value="Enter a line and odds to see fitted P(over), historical hit-rate, and EV.")
        ttk.Label(probe, textvariable=self.charts_probe_result,
                  foreground="#333", anchor="w", justify="left").grid(
            row=1, column=0, columnspan=5, padx=6, pady=4, sticky="we")

        # --- Summary text panel (always visible) --------------------------
        summary_frame = ttk.LabelFrame(
            outer, text="Book lines + P(over) + EV summary")
        summary_frame.pack(fill="x", pady=(8, 0))
        self.charts_summary = scrolledtext.ScrolledText(
            summary_frame, height=10, wrap="none", state="disabled",
            font=("Menlo", 10),
        )
        self.charts_summary.pack(fill="x")

        self._charts_top_canvas = None
        self._charts_bottom_canvas = None
        self._charts_splits_canvas = None
        self._charts_hitrate_canvas = None
        self._charts_player_index: list[dict] = []
        self._charts_last_data = None  # last fetched PlayerChartData

        try:
            self._charts_reload_lookups()
        except Exception:
            pass

    def _charts_reload_lookups(self) -> None:
        try:
            teams = self._player_charts_mod.list_teams(
                self.charts_db_path.get().strip())
        except Exception as exc:
            messagebox.showerror("Charts", f"Failed to read DB: {exc}")
            return
        values = ["(any)"] + teams
        self.charts_team_box.configure(values=values)
        if self.charts_team.get() not in values:
            self.charts_team.set("(any)")
        self._charts_refresh_player_box()

    def _charts_refresh_player_box(self) -> None:
        team = self.charts_team.get().strip()
        if team in {"", "(any)", "any", "all"}:
            team_arg = None
        else:
            team_arg = team
        try:
            df = self._player_charts_mod.list_players_with_data(
                self.charts_db_path.get().strip(), team=team_arg)
        except Exception as exc:
            messagebox.showerror("Charts", f"Failed to list players: {exc}")
            return
        names = df["player_name"].astype(str).tolist()
        self._charts_player_index = df.to_dict("records")
        self.charts_player_box.configure(values=names)
        if names and self.charts_player.get() not in names:
            self.charts_player.set(names[0])

    def _charts_resolve_player_id(self, raw_name: str) -> tuple[int, str]:
        """Look up player_id, preferring DB matches before nba_api static list."""
        text = (raw_name or "").strip()
        if not text:
            raise ValueError("Player name is required")
        for rec in self._charts_player_index:
            if str(rec.get("player_name", "")).lower() == text.lower():
                return int(rec["player_id"]), str(rec["player_name"])
        # Fall back to existing resolver (uses nba_api static list).
        resolved = self._resolve_player_identity(text, allow_synthetic=False)
        return int(resolved["player_id"]), str(resolved["player_name"])

    def _charts_selected_distributions(self) -> tuple[str, ...]:
        sel: list[str] = []
        if self.charts_dist_normal.get():
            sel.append("normal")
        if self.charts_dist_poisson.get():
            sel.append("poisson")
        if self.charts_dist_negbin.get():
            sel.append("negative_binomial")
        return tuple(sel) if sel else ("normal",)

    def _charts_render(self) -> None:
        try:
            db_path = self.charts_db_path.get().strip() or "data/database/nba_data.db"
            player_id, player_name = self._charts_resolve_player_id(
                self.charts_player.get())
            stat_type = self.charts_stat.get().strip().lower() or "points"
            n_games = self._parse_int_or_default(self.charts_n_games.get(), 25)
            rolling = max(1, self._parse_int_or_default(
                self.charts_rolling.get(), 5))
            distributions = self._charts_selected_distributions()

            data = self._player_charts_mod.fetch_player_chart_data(
                db_path=db_path,
                player_id=player_id,
                player_name=player_name,
                stat_type=stat_type,
                n_games=n_games,
            )
            self._charts_last_data = data

            top_fig = self._player_charts_mod.build_recent_games_figure(
                data, rolling_window=rolling)
            dist_fig = self._player_charts_mod.build_distribution_figure(
                data, distributions=distributions)
            splits_fig = self._player_charts_mod.build_splits_figure(data)
            hit_fig = self._player_charts_mod.build_hit_rate_figure(data)
            summary = self._player_charts_mod.book_lines_summary_text(data)
        except Exception as exc:
            messagebox.showerror("Charts", str(exc))
            return

        self._charts_swap_figure(self._charts_top_holder,
                                 "_charts_top_canvas", top_fig)
        self._charts_swap_figure(self._charts_bottom_holder,
                                 "_charts_bottom_canvas", dist_fig)
        self._charts_swap_figure(self._charts_splits_holder,
                                 "_charts_splits_canvas", splits_fig)
        self._charts_swap_figure(self._charts_hitrate_holder,
                                 "_charts_hitrate_canvas", hit_fig)

        self.charts_summary.configure(state="normal")
        self.charts_summary.delete("1.0", tk.END)
        self.charts_summary.insert(tk.END, summary)
        self.charts_summary.configure(state="disabled")

        # Pre-populate custom-line probe with the market median if empty.
        if not self.charts_custom_line.get().strip() and data.market_median_line:
            self.charts_custom_line.set(f"{data.market_median_line:.1f}")
        self._charts_evaluate_custom(silent_if_empty=True)

    def _charts_evaluate_custom(self, silent_if_empty: bool = False) -> None:
        data = self._charts_last_data
        if data is None:
            if not silent_if_empty:
                messagebox.showinfo(
                    "Custom line probe",
                    "Click 'Show charts' first to load player data.",
                )
            return
        line_text = self.charts_custom_line.get().strip()
        odds_text = self.charts_custom_odds.get().strip()
        if not line_text:
            if silent_if_empty:
                return
            messagebox.showerror("Custom line probe", "Enter a line value.")
            return
        try:
            line_val = float(line_text)
        except ValueError:
            messagebox.showerror("Custom line probe", f"Invalid line: {line_text}")
            return
        odds_val: int | None = None
        if odds_text and odds_text.lower() not in {"none", "na", "-"}:
            try:
                odds_val = int(round(float(odds_text)))
            except ValueError:
                messagebox.showerror(
                    "Custom line probe", f"Invalid odds: {odds_text}")
                return

        result = self._player_charts_mod.evaluate_custom_line(
            data, line_val, american_odds=odds_val)
        p = result["p_over"]
        ev_o = result["ev_over_per_unit"]
        ev_u = result["ev_under_per_unit"]
        text = (
            f"line={result['line']:.1f}  |  fitted P(over)={p:.1%}"
            if p is not None else f"line={result['line']:.1f}  |  P(over)=n/a"
        )
        text += (
            f"  |  historical over={result['historical_over_rate']:.0%} "
            f"({result['hits']}/{result['n']} games)"
        )
        if ev_o is not None and ev_u is not None:
            text += f"\nEV per unit:  OVER {ev_o:+.3f}   UNDER {ev_u:+.3f}"
            best = "OVER" if ev_o > ev_u else "UNDER"
            best_ev = max(ev_o, ev_u)
            verdict = "+EV" if best_ev > 0 else "-EV"
            text += f"   ->  best={best} ({verdict} {best_ev:+.3f})"
        elif odds_val is None:
            text += "\n(provide American odds to compute EV)"
        self.charts_probe_result.set(text)

    def _charts_swap_figure(self, holder, canvas_attr: str, figure) -> None:
        old = getattr(self, canvas_attr, None)
        if old is not None:
            try:
                old.get_tk_widget().destroy()
            except Exception:
                pass
        canvas = self._mpl_canvas_cls(figure, master=holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        setattr(self, canvas_attr, canvas)

    # ---- Operations tab ----------------------------------------------------

    def _build_operations_tab(self, tab):
        """Build the Operations tab: launches every CLI pipeline."""
        outer = ttk.Frame(tab)
        outer.pack(fill="both", expand=True, padx=8, pady=8)

        # Top: notebook of operation groups so the UI doesn't sprawl.
        ops_nb = ttk.Notebook(outer)
        ops_nb.pack(fill="both", expand=True)

        etl_frame = ttk.Frame(ops_nb)
        web_frame = ttk.Frame(ops_nb)
        parser_frame = ttk.Frame(ops_nb)
        eval_frame = ttk.Frame(ops_nb)
        revx_frame = ttk.Frame(ops_nb)
        ops_nb.add(etl_frame, text="Daily ETL")
        ops_nb.add(web_frame, text="Web Text")
        ops_nb.add(parser_frame, text="Browser Parser")
        ops_nb.add(eval_frame, text="Evaluation")
        ops_nb.add(revx_frame, text="Reverse-Engineering")

        # Shared output panel + control buttons.
        bottom = ttk.Frame(outer)
        bottom.pack(fill="both", expand=True, pady=(8, 0))
        controls_row = ttk.Frame(bottom)
        controls_row.pack(fill="x")
        ttk.Label(controls_row, text="Output:").pack(side="left")
        self.ops_status_var = tk.StringVar(value="idle")
        ttk.Label(controls_row, textvariable=self.ops_status_var,
                  foreground="#555").pack(side="left", padx=(8, 0))
        ttk.Button(controls_row, text="Stop",
                   command=self._ops_stop).pack(side="right", padx=(4, 0))
        ttk.Button(controls_row, text="Clear",
                   command=self._ops_clear).pack(side="right")

        self.ops_output = scrolledtext.ScrolledText(
            bottom, height=18, wrap="word", state="disabled",
            font=("Menlo", 10),
        )
        self.ops_output.pack(fill="both", expand=True, pady=(4, 0))

        self.ops_runner = _OperationRunner(self.root, self.ops_output)

        self._build_ops_etl(etl_frame)
        self._build_ops_web(web_frame)
        self._build_ops_parser(parser_frame)
        self._build_ops_eval(eval_frame)
        self._build_ops_revx(revx_frame)

    @staticmethod
    def _ops_field(parent, row, label, default="", width=46):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=6, pady=3)
        var = tk.StringVar(value=str(default))
        ttk.Entry(parent, textvariable=var, width=width).grid(
            row=row, column=1, sticky="we", padx=6, pady=3)
        return var

    @staticmethod
    def _ops_check(parent, row, label, default=False):
        var = tk.BooleanVar(value=bool(default))
        ttk.Checkbutton(parent, text=label, variable=var).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6, pady=2)
        return var

    def _ops_browse_file(self, var: tk.StringVar) -> None:
        chosen = filedialog.askopenfilename()
        if chosen:
            var.set(chosen)

    def _ops_browse_save(self, var: tk.StringVar) -> None:
        chosen = filedialog.asksaveasfilename()
        if chosen:
            var.set(chosen)

    def _ops_browse_dir(self, var: tk.StringVar) -> None:
        chosen = filedialog.askdirectory()
        if chosen:
            var.set(chosen)

    def _build_ops_etl(self, frame):
        frame.columnconfigure(1, weight=1)
        self.etl_db_path = self._ops_field(
            frame, 0, "DB path", "data/database/nba_data.db")
        self.etl_season = self._ops_field(frame, 1, "Season (e.g. 2025-26)", "")
        self.etl_players = self._ops_field(frame, 2, "Players (space-separated)", "")
        self.etl_min_players = self._ops_field(frame, 3, "Min players", "0")
        self.etl_player_limit = self._ops_field(frame, 4, "Player limit", "")
        self.etl_game_log_games = self._ops_field(frame, 5, "Game-log games", "")

        flags = ttk.LabelFrame(frame, text="Skip flags / behavior")
        flags.grid(row=6, column=0, columnspan=2, sticky="we", padx=4, pady=6)
        self.etl_skip_game_logs = self._ops_check(flags, 0, "skip game logs")
        self.etl_skip_team_defense = self._ops_check(flags, 1, "skip team defense")
        self.etl_skip_odds = self._ops_check(flags, 2, "skip odds")
        self.etl_skip_browser_parser = self._ops_check(
            flags, 3, "skip browser parser")
        self.etl_skip_reverse_engineering = self._ops_check(
            flags, 4, "skip reverse-engineering")
        self.etl_all_db_players = self._ops_check(flags, 5, "all DB players")
        self.etl_skip_zero_games = self._ops_check(
            flags, 6, "skip zero-game players")
        self.etl_strict = self._ops_check(flags, 7, "strict mode")
        self.etl_force_odds = self._ops_check(flags, 8, "force odds poll")
        self.etl_reverse_single_pass = self._ops_check(
            flags, 9, "reverse-engineering single pass")

        web = ttk.LabelFrame(frame, text="Web-text + browser session")
        web.grid(row=7, column=0, columnspan=2, sticky="we", padx=4, pady=6)
        web.columnconfigure(1, weight=1)
        self.etl_web_urls_file = self._ops_field(web, 0, "Web URLs file", "")
        ttk.Button(web, text="Browse",
                   command=lambda: self._ops_browse_file(self.etl_web_urls_file)
                   ).grid(row=0, column=2, padx=4)
        self.etl_web_urls_inline = self._ops_field(
            web, 1, "Web URLs (space-separated)", "")
        self.etl_browser_auth = self._ops_field(
            web, 2, "Browser auth state file", "")
        ttk.Button(web, text="Browse",
                   command=lambda: self._ops_browse_file(self.etl_browser_auth)
                   ).grid(row=2, column=2, padx=4)
        self.etl_chrome_port = self._ops_field(
            web, 3, "Chrome debug port (e.g. 9222)", "")
        self.etl_login_url = self._ops_field(
            web, 4, "Login URL (optional)", "")
        self.etl_validate_first = self._ops_check(
            web, 5, "validate session before ETL")
        self.etl_force_web_poll = self._ops_check(
            web, 6, "force web-text poll")

        ttk.Button(frame, text="Run Daily ETL",
                   command=self._ops_run_daily_etl).grid(
            row=8, column=0, sticky="w", padx=6, pady=10)

    def _build_ops_web(self, frame):
        frame.columnconfigure(1, weight=1)
        self.web_url = self._ops_field(
            frame, 0, "Target URL",
            "https://app.prizepicks.com/board/nba",
        )
        self.web_auth_file = self._ops_field(
            frame, 1, "Auth state file",
            "data/config/auth/prizepicks_state.json",
        )
        ttk.Button(frame, text="Browse",
                   command=lambda: self._ops_browse_file(self.web_auth_file)
                   ).grid(row=1, column=2, padx=4)
        self.web_chrome_port = self._ops_field(
            frame, 2, "Chrome debug port", "9222")
        self.web_user_data_dir = self._ops_field(
            frame, 3, "User data dir (optional)", "")
        self.web_active_players_out = self._ops_field(
            frame, 4, "Active-players output file",
            "data/config/active_nba_players.txt",
        )

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=5, column=0, columnspan=3, sticky="w", padx=4, pady=10)
        ttk.Button(btn_row, text="Login (headed)",
                   command=self._ops_run_login).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Validate Session",
                   command=self._ops_run_validate).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Fetch URL (force-poll)",
                   command=self._ops_run_fetch_url).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Connect Chrome (CDP capture)",
                   command=self._ops_run_connect_chrome).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Extract Chrome Cookies",
                   command=self._ops_run_extract_chrome).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Sync Active Players Ref",
                   command=self._ops_run_sync_players).pack(side="left", padx=4)

    def _build_ops_parser(self, frame):
        frame.columnconfigure(1, weight=1)
        self.parser_urls_file = self._ops_field(
            frame, 0, "URLs file", "data/config/web_text_urls.txt")
        ttk.Button(frame, text="Browse",
                   command=lambda: self._ops_browse_file(self.parser_urls_file)
                   ).grid(row=0, column=2, padx=4)
        self.parser_min_conf = self._ops_field(
            frame, 1, "Min parse confidence", "0.50")
        self.parser_max_per_url = self._ops_field(
            frame, 2, "Max snapshots per URL", "2")
        ttk.Button(frame, text="Run Browser Prop Parser",
                   command=self._ops_run_browser_parser).grid(
            row=3, column=0, sticky="w", padx=6, pady=10)

    def _build_ops_eval(self, frame):
        frame.columnconfigure(1, weight=1)
        self.eval_start_date = self._ops_field(
            frame, 0, "Start date (YYYY-MM-DD)", "")
        self.eval_end_date = self._ops_field(
            frame, 1, "End date (YYYY-MM-DD)", "")
        self.eval_stat_types = self._ops_field(
            frame, 2, "Stat types (space-sep)", "points assists rebounds pra")
        self.eval_windows = self._ops_field(
            frame, 3, "Windows (space-sep)", "5 7 10 15")
        self.eval_distributions = self._ops_field(
            frame, 4, "Distributions", "normal")
        self.eval_edge_threshold = self._ops_field(
            frame, 5, "Line-comparison edge threshold", "0.02")

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=6, column=0, columnspan=2, sticky="w", padx=4, pady=10)
        ttk.Button(btn_row, text="Real-data Benchmark",
                   command=self._ops_run_real_benchmark).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Distribution Sweep",
                   command=self._ops_run_distribution_sweep).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Line Comparison",
                   command=self._ops_run_line_comparison).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Monthly Diagnostics",
                   command=self._ops_run_monthly_diag).pack(side="left", padx=4)

    def _build_ops_revx(self, frame):
        frame.columnconfigure(1, weight=1)
        self.revx_source = self._ops_field(
            frame, 0, "Source (book/predictions/both)", "both")
        self.revx_poll_seconds = self._ops_field(
            frame, 1, "Poll seconds", "300")
        self.revx_min_inferred_rows = self._ops_field(
            frame, 2, "Min inferred rows", "25")
        self.revx_min_book_groups = self._ops_field(
            frame, 3, "Min book/stat groups", "2")
        self.revx_min_player_segments = self._ops_field(
            frame, 4, "Min player-segment groups", "5")
        self.revx_stability_runs = self._ops_field(
            frame, 5, "Require stability runs", "2")
        self.revx_stability_tol = self._ops_field(
            frame, 6, "Stability tolerance", "0.10")
        self.revx_continuous = self._ops_check(frame, 7, "continuous mode", True)

        ttk.Button(frame, text="Run Reverse-Engineering",
                   command=self._ops_run_reverse_engineering).grid(
            row=8, column=0, sticky="w", padx=6, pady=10)

    # ---- Operations: helpers and command launchers --------------------------

    @staticmethod
    def _split_ws(value: str) -> list[str]:
        return [tok for tok in (value or "").split() if tok]

    @staticmethod
    def _flag_if(args: list[str], flag: str, condition: bool) -> None:
        if condition:
            args.append(flag)

    @staticmethod
    def _opt_if(args: list[str], flag: str, value) -> None:
        text = str(value if value is not None else "").strip()
        if text != "":
            args.extend([flag, text])

    def _ops_launch(self, label: str, module: str, extra_args: list[str]) -> None:
        cmd = [sys.executable, "-m", module, *extra_args]
        cwd = str(Path(__file__).resolve().parents[1])
        self.ops_status_var.set(f"running: {label}")
        ok = self.ops_runner.start(
            label=label, cmd_args=cmd, cwd=cwd,
            on_done=lambda rc: self.ops_status_var.set(
                f"idle (last exit {rc})"),
        )
        if not ok:
            self.ops_status_var.set("idle")

    def _ops_stop(self) -> None:
        self.ops_runner.stop()

    def _ops_clear(self) -> None:
        self.ops_runner.clear()

    # --- ETL launcher
    def _ops_run_daily_etl(self) -> None:
        args: list[str] = []
        self._opt_if(args, "--db-path", self.etl_db_path.get())
        self._opt_if(args, "--season", self.etl_season.get())
        players = self._split_ws(self.etl_players.get())
        if players:
            args.extend(["--players", *players])
        self._opt_if(args, "--min-players", self.etl_min_players.get())
        self._opt_if(args, "--player-limit", self.etl_player_limit.get())
        self._opt_if(args, "--game-log-games", self.etl_game_log_games.get())
        self._flag_if(args, "--skip-game-logs", self.etl_skip_game_logs.get())
        self._flag_if(args, "--skip-team-defense",
                      self.etl_skip_team_defense.get())
        self._flag_if(args, "--skip-odds", self.etl_skip_odds.get())
        self._flag_if(args, "--skip-browser-parser",
                      self.etl_skip_browser_parser.get())
        self._flag_if(args, "--skip-reverse-engineering",
                      self.etl_skip_reverse_engineering.get())
        self._flag_if(args, "--all-db-players",
                      self.etl_all_db_players.get())
        self._flag_if(args, "--skip-zero-game-players",
                      self.etl_skip_zero_games.get())
        self._flag_if(args, "--strict", self.etl_strict.get())
        self._flag_if(args, "--force-odds-poll", self.etl_force_odds.get())
        self._flag_if(args, "--reverse-engineering-single-pass",
                      self.etl_reverse_single_pass.get())
        self._opt_if(args, "--web-text-urls-file",
                     self.etl_web_urls_file.get())
        urls = self._split_ws(self.etl_web_urls_inline.get())
        if urls:
            args.extend(["--web-text-urls", *urls])
        self._opt_if(args, "--browser-auth-state-file",
                     self.etl_browser_auth.get())
        self._opt_if(args, "--chrome-debug-port",
                     self.etl_chrome_port.get())
        self._opt_if(args, "--login", self.etl_login_url.get())
        self._flag_if(args, "--validate-session-before-etl",
                      self.etl_validate_first.get())
        self._flag_if(args, "--web-text-force-poll",
                      self.etl_force_web_poll.get())
        self._ops_launch("Daily ETL", "nba_model.data.daily_etl", args)

    # --- Web text launchers
    def _ops_run_login(self) -> None:
        url = self.web_url.get().strip()
        if not url:
            messagebox.showerror("Login", "Target URL is required.")
            return
        args = ["--login", url]
        self._opt_if(args, "--browser-auth-state-file",
                     self.web_auth_file.get())
        self._opt_if(args, "--browser-user-data-dir",
                     self.web_user_data_dir.get())
        self._ops_launch("Login (headed)",
                         "nba_model.model.web_text_ingestion", args)

    def _ops_run_validate(self) -> None:
        url = self.web_url.get().strip()
        if not url:
            messagebox.showerror("Validate", "Target URL is required.")
            return
        args = ["--validate-session", url]
        self._opt_if(args, "--browser-auth-state-file",
                     self.web_auth_file.get())
        self._opt_if(args, "--browser-user-data-dir",
                     self.web_user_data_dir.get())
        self._opt_if(args, "--chrome-debug-port",
                     self.web_chrome_port.get())
        self._ops_launch("Validate Session",
                         "nba_model.model.web_text_ingestion", args)

    def _ops_run_fetch_url(self) -> None:
        url = self.web_url.get().strip()
        if not url:
            messagebox.showerror("Fetch", "Target URL is required.")
            return
        args = ["--urls", url, "--force-poll"]
        self._opt_if(args, "--browser-auth-state-file",
                     self.web_auth_file.get())
        self._opt_if(args, "--browser-user-data-dir",
                     self.web_user_data_dir.get())
        self._opt_if(args, "--chrome-debug-port",
                     self.web_chrome_port.get())
        self._ops_launch("Fetch URL",
                         "nba_model.model.web_text_ingestion", args)

    def _ops_run_connect_chrome(self) -> None:
        url = self.web_url.get().strip()
        if not url:
            messagebox.showerror("Connect Chrome", "Target URL is required.")
            return
        args = ["--connect-chrome", url]
        self._opt_if(args, "--browser-auth-state-file",
                     self.web_auth_file.get())
        self._opt_if(args, "--chrome-debug-port",
                     self.web_chrome_port.get())
        self._ops_launch("Connect Chrome (CDP capture)",
                         "nba_model.model.web_text_ingestion", args)

    def _ops_run_extract_chrome(self) -> None:
        url = self.web_url.get().strip()
        if not url:
            messagebox.showerror(
                "Extract Chrome Session", "Target URL is required.")
            return
        args = ["--extract-chrome-session", url]
        self._opt_if(args, "--browser-auth-state-file",
                     self.web_auth_file.get())
        self._ops_launch("Extract Chrome Cookies",
                         "nba_model.model.web_text_ingestion", args)

    def _ops_run_sync_players(self) -> None:
        args = ["--sync-active-players-ref"]
        self._opt_if(args, "--active-players-output-file",
                     self.web_active_players_out.get())
        self._ops_launch("Sync Active Players Ref",
                         "nba_model.model.web_text_ingestion", args)

    # --- Browser prop parser
    def _ops_run_browser_parser(self) -> None:
        args: list[str] = []
        self._opt_if(args, "--urls-file", self.parser_urls_file.get())
        self._opt_if(args, "--min-parse-confidence", self.parser_min_conf.get())
        self._opt_if(args, "--max-snapshots-per-url",
                     self.parser_max_per_url.get())
        self._ops_launch("Browser Prop Parser",
                         "nba_model.model.browser_prop_parser", args)

    # --- Evaluation launchers
    def _eval_common_args(self) -> list[str]:
        args: list[str] = []
        self._opt_if(args, "--start-date", self.eval_start_date.get())
        self._opt_if(args, "--end-date", self.eval_end_date.get())
        stats = self._split_ws(self.eval_stat_types.get())
        if stats:
            args.extend(["--stat-types", *stats])
        return args

    def _ops_run_real_benchmark(self) -> None:
        args = self._eval_common_args()
        windows = self._split_ws(self.eval_windows.get())
        if windows:
            args.extend(["--windows", *windows])
        dists = self._split_ws(self.eval_distributions.get())
        if dists:
            args.extend(["--distributions", *dists])
        self._ops_launch("Real-data Benchmark",
                         "nba_model.evaluation.run_real_data_benchmark", args)

    def _ops_run_distribution_sweep(self) -> None:
        args = self._eval_common_args()
        windows = self._split_ws(self.eval_windows.get())
        if windows:
            args.extend(["--windows", *windows])
        self._ops_launch("Distribution Sweep",
                         "nba_model.evaluation.run_distribution_sweep", args)

    def _ops_run_line_comparison(self) -> None:
        args = self._eval_common_args()
        self._opt_if(args, "--edge-threshold", self.eval_edge_threshold.get())
        self._ops_launch("Line Comparison",
                         "nba_model.evaluation.line_comparison", args)

    def _ops_run_monthly_diag(self) -> None:
        args = self._eval_common_args()
        self._ops_launch("Monthly Diagnostics",
                         "nba_model.evaluation.monthly_diagnostics", args)

    # --- Reverse engineering
    def _ops_run_reverse_engineering(self) -> None:
        args: list[str] = []
        self._opt_if(args, "--source", self.revx_source.get())
        self._opt_if(args, "--poll-seconds", self.revx_poll_seconds.get())
        self._opt_if(args, "--min-inferred-rows",
                     self.revx_min_inferred_rows.get())
        self._opt_if(args, "--min-book-stat-groups",
                     self.revx_min_book_groups.get())
        self._opt_if(args, "--min-player-segment-groups",
                     self.revx_min_player_segments.get())
        self._opt_if(args, "--require-stability-runs",
                     self.revx_stability_runs.get())
        self._opt_if(args, "--stability-tolerance",
                     self.revx_stability_tol.get())
        self._flag_if(args, "--continuous", self.revx_continuous.get())
        self._ops_launch("Reverse-Engineering",
                         "nba_model.evaluation.market_reverse_engineering",
                         args)

    def _reset_single_tuning_defaults(self):
        """Reset single-prop tuning sliders back to model defaults."""
        self.single_distribution.set(DEFAULT_SINGLE_PROP_DISTRIBUTION)
        self.single_league_avg_def.set(float(DEFAULT_LEAGUE_AVG_DEF_RATING))
        self.single_defense_sensitivity.set(float(DEFAULT_DEFENSE_SENSITIVITY))
        self.single_blowout_threshold.set(float(DEFAULT_BLOWOUT_THRESHOLD))
        self.single_blowout_penalty.set(float(DEFAULT_BLOWOUT_PENALTY))
        self.single_defense_severity.set(1.0)
        self.single_minutes_severity.set(1.0)
        self.single_sigma_severity.set(1.0)

    def _reset_parlay_tuning_defaults(self):
        """Reset parlay tuning sliders back to baseline defaults."""
        self.parlay_corr_severity.set(1.0)
        self.parlay_vol_severity.set(1.0)

    def _run_single(self):
        try:
            result = run_single_prop(
                player_name=self.single_player.get().strip(),
                line=self._parse_float_or_default(
                    self.single_line.get(), DEFAULT_POINTS_LINE),
                rolling_window=self._parse_int_or_default(
                    self.single_window.get(), DEFAULT_ROLLING_WINDOW),
                american_odds=self._parse_int_or_default(
                    self.single_odds.get(), DEFAULT_AMERICAN_ODDS),
                opp_def_rating=self._parse_float_or_default(
                    self.single_opp_def.get(), DEFAULT_OPP_DEF_RATING),
                vegas_spread=self._parse_float_or_default(
                    self.single_spread.get(), DEFAULT_VEGAS_SPREAD),
                league_avg_def_rating=float(self.single_league_avg_def.get()),
                defense_sensitivity=float(
                    self.single_defense_sensitivity.get()),
                blowout_threshold=float(self.single_blowout_threshold.get()),
                blowout_penalty=float(self.single_blowout_penalty.get()),
                n_games=self._parse_int_or_default(
                    self.single_n_games.get(), 100),
                show_plot=self.single_show_plot.get(),
                distribution=self.single_distribution.get().strip().lower(),
                defense_severity=float(self.single_defense_severity.get()),
                minutes_penalty_severity=float(
                    self.single_minutes_severity.get()),
                sigma_severity=float(self.single_sigma_severity.get()),
            )
            self._render_output(self.single_output, result)
        except Exception as exc:
            messagebox.showerror("Single Prop Error", str(exc))

    def _run_parlay(self):
        try:
            stats = _normalize_parlay_stats(
                self._parse_csv_strings(self.parlay_stats.get()))
            lines = self._parse_csv_floats(self.parlay_lines.get())
            if len(stats) != len(lines):
                raise ValueError(
                    "Parlay stats count must match parlay lines count.")

            sportsbook = self.parlay_sportsbook.get().strip().lower()
            odds_format = self.parlay_odds_format.get().strip().lower()
            odds_value = self._parse_float_or_default(
                self.parlay_odds.get(), 3.0)
            american_odds = _resolve_parlay_american_odds(
                parlay_odds=odds_value,
                parlay_odds_format=odds_format,
                fallback_american_odds=DEFAULT_AMERICAN_ODDS,
            )

            result = run_parlay_demo(
                player_name=self.parlay_player.get().strip(),
                stats_cols=stats,
                lines=lines,
                american_odds=american_odds,
                sportsbook=sportsbook,
                n_games=self._parse_int_or_default(
                    self.parlay_n_games.get(), 100),
                n_sims=self._parse_int_or_default(
                    self.parlay_n_sims.get(), 20000),
                correlation_severity=float(self.parlay_corr_severity.get()),
                volatility_severity=float(self.parlay_vol_severity.get()),
            )
            self._render_output(self.parlay_output, result)
        except Exception as exc:
            messagebox.showerror("Parlay Error", str(exc))


def main() -> None:
    """Launch the desktop UI."""
    root = tk.Tk()
    app = SimpleModelUI(root)
    del app
    root.mainloop()


if __name__ == "__main__":
    main()
