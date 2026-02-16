import csv
import json
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from nba_api.stats.static import players as static_players

# Allow direct execution via ".../nba_model/simple_ui.py"
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from nba_model.data.database.db_manager import DatabaseManager  # noqa: E402
from nba_model.run_model import (  # noqa: E402
    DEFAULT_AMERICAN_ODDS,
    DEFAULT_OPP_DEF_RATING,
    DEFAULT_PARLAY_LINES,
    DEFAULT_PARLAY_STATS,
    DEFAULT_PLAYER_NAME,
    DEFAULT_POINTS_LINE,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_VEGAS_SPREAD,
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


class SimpleModelUI:
    """Small desktop UI to run single-prop and parlay workflows."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NBA Props - Simple UI")
        self.root.geometry("900x650")
        self.manual_records = []
        self._player_lookup_cache = {}

        self._build_layout()

    def _build_layout(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        single_tab = ttk.Frame(notebook)
        parlay_tab = ttk.Frame(notebook)
        manual_tab = ttk.Frame(notebook)
        notebook.add(single_tab, text="Single Prop")
        notebook.add(parlay_tab, text="Parlay (PrizePicks/Underdog)")
        notebook.add(manual_tab, text="Manual Lines Import")

        self._build_single_tab(single_tab)
        self._build_parlay_tab(parlay_tab)
        self._build_manual_tab(manual_tab)

    @staticmethod
    def _add_labeled_entry(parent, row, label, default):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=6)
        var = tk.StringVar(value=str(default))
        entry = ttk.Entry(parent, textvariable=var, width=28)
        entry.grid(row=row, column=1, sticky="w", padx=6, pady=6)
        return var

    @staticmethod
    def _parse_csv_strings(value: str):
        return [v.strip() for v in value.split(",") if v.strip()]

    @staticmethod
    def _parse_csv_floats(value: str):
        values = [v.strip() for v in value.split(",") if v.strip()]
        return [float(v) for v in values]

    @staticmethod
    def _canonical_stat_key(value: str) -> str:
        return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())

    def _normalize_stat_type(self, value: str) -> str:
        key = self._canonical_stat_key(value)
        stat_type = _MANUAL_STAT_ALIASES.get(key)
        if not stat_type:
            supported = sorted(set(_MANUAL_STAT_ALIASES.values()))
            raise ValueError(f"Unsupported stat '{value}'. Supported: {supported}")
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
            raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD format.") from exc

    def _resolve_player_identity(self, player_name: str) -> dict:
        raw_name = str(player_name).strip()
        if not raw_name:
            raise ValueError("Player name is required")
        if raw_name in self._player_lookup_cache:
            return self._player_lookup_cache[raw_name]

        candidates = static_players.find_players_by_full_name(raw_name)
        if not candidates:
            candidates = static_players.find_players_by_full_name(raw_name.replace(".", ""))
        if not candidates:
            raise ValueError(f"Player '{raw_name}' not found in NBA player list")

        exact = next((c for c in candidates if c.get("full_name", "").lower() == raw_name.lower()), candidates[0])
        resolved = {"player_id": int(exact["id"]), "player_name": exact["full_name"]}
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

        player = self._resolve_player_identity(player_name)
        stat_type = self._normalize_stat_type(stat_token)
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

    def _parse_manual_lines_text(self, text: str, default_game_date: str, default_book: str):
        records = []
        errors = []
        normalized_date = self._normalize_game_date(default_game_date)
        normalized_book = (default_book or "").strip() or "manual_ui"

        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
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
        return records, errors

    def _render_output(self, widget: scrolledtext.ScrolledText, payload):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, json.dumps(payload, indent=2))
        widget.configure(state="disabled")

    def _build_single_tab(self, tab):
        controls = ttk.Frame(tab)
        controls.pack(fill="x", padx=8, pady=8)

        self.single_player = self._add_labeled_entry(controls, 0, "Player", DEFAULT_PLAYER_NAME)
        self.single_line = self._add_labeled_entry(controls, 1, "Line", DEFAULT_POINTS_LINE)
        self.single_odds = self._add_labeled_entry(controls, 2, "American Odds", DEFAULT_AMERICAN_ODDS)
        self.single_window = self._add_labeled_entry(controls, 3, "Rolling Window", DEFAULT_ROLLING_WINDOW)
        self.single_opp_def = self._add_labeled_entry(controls, 4, "Opponent Def Rating", DEFAULT_OPP_DEF_RATING)
        self.single_spread = self._add_labeled_entry(controls, 5, "Vegas Spread", DEFAULT_VEGAS_SPREAD)
        self.single_n_games = self._add_labeled_entry(controls, 6, "History Games", 100)

        self.single_show_plot = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Show Distribution Plot", variable=self.single_show_plot).grid(
            row=7, column=0, sticky="w", padx=6, pady=6
        )

        ttk.Button(controls, text="Run Single Prop", command=self._run_single).grid(
            row=7, column=1, sticky="w", padx=6, pady=6
        )

        self.single_output = scrolledtext.ScrolledText(tab, height=18, wrap=tk.WORD, state="disabled")
        self.single_output.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _build_parlay_tab(self, tab):
        controls = ttk.Frame(tab)
        controls.pack(fill="x", padx=8, pady=8)

        self.parlay_player = self._add_labeled_entry(controls, 0, "Player", DEFAULT_PLAYER_NAME)

        ttk.Label(controls, text="Sportsbook").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self.parlay_sportsbook = tk.StringVar(value="prizepicks")
        sportsbook_box = ttk.Combobox(
            controls,
            textvariable=self.parlay_sportsbook,
            values=["custom", "prizepicks", "underdog"],
            state="readonly",
            width=25,
        )
        sportsbook_box.grid(row=1, column=1, sticky="w", padx=6, pady=6)

        self.parlay_stats = self._add_labeled_entry(
            controls, 2, "Leg Stats (csv)", ", ".join(DEFAULT_PARLAY_STATS)
        )
        self.parlay_lines = self._add_labeled_entry(
            controls, 3, "Leg Lines (csv)", ", ".join(str(x) for x in DEFAULT_PARLAY_LINES)
        )

        ttk.Label(controls, text="Parlay Odds Format").grid(row=4, column=0, sticky="w", padx=6, pady=6)
        self.parlay_odds_format = tk.StringVar(value="multiplier")
        odds_format_box = ttk.Combobox(
            controls,
            textvariable=self.parlay_odds_format,
            values=["american", "decimal", "multiplier"],
            state="readonly",
            width=25,
        )
        odds_format_box.grid(row=4, column=1, sticky="w", padx=6, pady=6)

        self.parlay_odds = self._add_labeled_entry(controls, 5, "Parlay Odds", 3.0)
        self.parlay_n_games = self._add_labeled_entry(controls, 6, "History Games", 100)
        self.parlay_n_sims = self._add_labeled_entry(controls, 7, "Simulation Runs", 20000)

        ttk.Button(controls, text="Run Parlay", command=self._run_parlay).grid(
            row=8, column=1, sticky="w", padx=6, pady=6
        )

        self.parlay_output = scrolledtext.ScrolledText(tab, height=16, wrap=tk.WORD, state="disabled")
        self.parlay_output.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _build_manual_tab(self, tab):
        controls = ttk.Frame(tab)
        controls.pack(fill="x", padx=8, pady=8)

        self.manual_default_date = self._add_labeled_entry(
            controls, 0, "Default Game Date (YYYY-MM-DD)", datetime.utcnow().strftime("%Y-%m-%d")
        )
        self.manual_default_book = self._add_labeled_entry(controls, 1, "Default Sportsbook", "manual_ui")
        self.manual_db_path = self._add_labeled_entry(controls, 2, "DB Path", "data/database/nba_data.db")

        ttk.Button(controls, text="Parse Lines", command=self._parse_manual_lines).grid(
            row=3, column=0, sticky="w", padx=6, pady=6
        )
        ttk.Button(controls, text="Save Parsed Lines to DB", command=self._save_manual_lines).grid(
            row=3, column=1, sticky="w", padx=6, pady=6
        )

        help_text = (
            "Paste one prop per line (pipe, csv, or tab-delimited).\n"
            "Formats:\n"
            "  1) player | stat | line | over_odds | under_odds\n"
            "  2) player | game_date | book | stat | line | over_odds | under_odds\n"
            "Notes: stats accept aliases (pts/ast/reb/pra). Odds are optional."
        )
        ttk.Label(tab, text=help_text, justify="left").pack(anchor="w", padx=8, pady=(0, 6))

        self.manual_input = scrolledtext.ScrolledText(tab, height=12, wrap=tk.WORD)
        self.manual_input.pack(fill="x", expand=False, padx=8, pady=(0, 8))
        self.manual_input.insert(
            tk.END,
            "# Example rows\n"
            "LeBron James | points | 27.5 | -115 | -105\n"
            "Stephen Curry | 2025-03-07 | FanDuel | assists | 5.5 | -110 | -120\n",
        )

        self.manual_output = scrolledtext.ScrolledText(tab, height=12, wrap=tk.WORD, state="disabled")
        self.manual_output.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _parse_manual_lines(self):
        try:
            text = self.manual_input.get("1.0", tk.END)
            records, errors = self._parse_manual_lines_text(
                text=text,
                default_game_date=self.manual_default_date.get().strip(),
                default_book=self.manual_default_book.get().strip(),
            )
            self.manual_records = records
            payload = {
                "parsed_records": len(records),
                "errors": errors,
                "sample_records": records[:15],
            }
            self._render_output(self.manual_output, payload)
            if errors:
                messagebox.showwarning(
                    "Manual Lines Parse",
                    f"Parsed {len(records)} line(s) with {len(errors)} error(s). Review output for details.",
                )
            elif records:
                messagebox.showinfo(
                    "Manual Lines Parse",
                    f"Parsed {len(records)} line(s). Click 'Save Parsed Lines to DB' to store them.",
                )
            else:
                messagebox.showinfo("Manual Lines Parse", "No valid lines were found in the input box.")
        except Exception as exc:
            messagebox.showerror("Manual Lines Parse Error", str(exc))

    def _save_manual_lines(self):
        try:
            if not self.manual_records:
                raise ValueError("No parsed lines available. Click 'Parse Lines' first.")

            db_path = self.manual_db_path.get().strip() or "data/database/nba_data.db"
            with DatabaseManager(db_path=db_path) as db:
                before_count = db.conn.execute("SELECT COUNT(*) FROM betting_lines").fetchone()[0]
                seen = {}
                for row in self.manual_records:
                    player_id = row["player_id"]
                    if player_id not in seen:
                        seen[player_id] = row["player_name"]
                        db.insert_player(player_id, row["player_name"])
                db.insert_betting_lines_records(self.manual_records)
                after_count = db.conn.execute("SELECT COUNT(*) FROM betting_lines").fetchone()[0]

            inserted = after_count - before_count
            payload = {
                "db_path": db_path,
                "parsed_rows_submitted": len(self.manual_records),
                "new_rows_inserted": inserted,
                "total_betting_lines_rows": after_count,
            }
            self._render_output(self.manual_output, payload)
            messagebox.showinfo(
                "Manual Lines Save",
                f"Saved {len(self.manual_records)} parsed row(s). Inserted {inserted} new betting_lines row(s).",
            )
        except Exception as exc:
            messagebox.showerror("Manual Lines Save Error", str(exc))

    def _run_single(self):
        try:
            result = run_single_prop(
                player_name=self.single_player.get().strip(),
                line=float(self.single_line.get()),
                rolling_window=int(self.single_window.get()),
                american_odds=int(float(self.single_odds.get())),
                opp_def_rating=float(self.single_opp_def.get()),
                vegas_spread=float(self.single_spread.get()),
                n_games=int(self.single_n_games.get()),
                show_plot=self.single_show_plot.get(),
            )
            self._render_output(self.single_output, result)
        except Exception as exc:
            messagebox.showerror("Single Prop Error", str(exc))

    def _run_parlay(self):
        try:
            stats = _normalize_parlay_stats(self._parse_csv_strings(self.parlay_stats.get()))
            lines = self._parse_csv_floats(self.parlay_lines.get())
            if len(stats) != len(lines):
                raise ValueError("Parlay stats count must match parlay lines count.")

            sportsbook = self.parlay_sportsbook.get().strip().lower()
            odds_format = self.parlay_odds_format.get().strip().lower()
            odds_value = float(self.parlay_odds.get())
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
                n_games=int(self.parlay_n_games.get()),
                n_sims=int(self.parlay_n_sims.get()),
            )
            self._render_output(self.parlay_output, result)
        except Exception as exc:
            messagebox.showerror("Parlay Error", str(exc))


def main():
    root = tk.Tk()
    app = SimpleModelUI(root)
    del app
    root.mainloop()


if __name__ == "__main__":
    main()
