import csv
import json
import re
import sys
import tkinter as tk
from datetime import datetime
from hashlib import sha1
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
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=6)
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
        ttk.Label(container, textvariable=value_text, width=7, anchor="e").pack(side="left", padx=(8, 0))
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
        slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        return slug or "custom_stat"

    def _normalize_stat_type(self, value: str, allow_custom: bool = False) -> str:
        key = self._canonical_stat_key(value)
        stat_type = _MANUAL_STAT_ALIASES.get(key)
        if not stat_type:
            if allow_custom:
                return self._slug_stat_type(value)
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

    @staticmethod
    def _synthetic_player_id(player_name: str) -> int:
        """
        Deterministic synthetic player id for non-NBA names.

        Keeps manual import usable for non-NBA boards while avoiding collisions
        with normal NBA IDs by using a high numeric range.
        """
        digest = sha1(player_name.strip().lower().encode("utf-8")).hexdigest()[:10]
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
            candidates = static_players.find_players_by_full_name(raw_name.replace(".", ""))
        if not candidates:
            if allow_synthetic:
                resolved = {"player_id": self._synthetic_player_id(raw_name), "player_name": raw_name}
                self._player_lookup_cache[raw_name] = resolved
                return resolved
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

        player = self._resolve_player_identity(player_name, allow_synthetic=True)
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
            if not stat_token or self._is_noise_line(stat_token) or self._is_matchup_line(stat_token):
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
                player_name = re.sub(r"(Goblin|Demon)+$", "", candidate, flags=re.IGNORECASE).strip()
                if player_name:
                    break
            if not player_name:
                continue

            try:
                player = self._resolve_player_identity(player_name, allow_synthetic=True)
                stat_type = self._normalize_stat_type(stat_token, allow_custom=True)
                line_value = float(line_value_token)
            except Exception as exc:
                errors.append(f"board parse near '{player_name}' / '{stat_token}': {exc}")
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
            has_structured_delimiter = ("|" in stripped) or ("\t" in stripped) or (stripped.count(",") >= 2)
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
                "No parseable props found. Use delimiter format or paste full board text that includes "
                "player + matchup + line + stat blocks."
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
            messagebox.showinfo("Manual Lines", "Select one or more parsed rows to delete.")
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

        self.single_player = self._add_labeled_entry(controls, 0, "Player", DEFAULT_PLAYER_NAME)
        self.single_line = self._add_labeled_entry(controls, 1, "Line", DEFAULT_POINTS_LINE)
        self.single_odds = self._add_labeled_entry(controls, 2, "American Odds", DEFAULT_AMERICAN_ODDS)
        self.single_window = self._add_labeled_entry(controls, 3, "Rolling Window", DEFAULT_ROLLING_WINDOW)
        self.single_opp_def = self._add_labeled_entry(controls, 4, "Opponent Def Rating", DEFAULT_OPP_DEF_RATING)
        self.single_spread = self._add_labeled_entry(controls, 5, "Vegas Spread", DEFAULT_VEGAS_SPREAD)
        self.single_n_games = self._add_labeled_entry(controls, 6, "History Games", 100)
        ttk.Label(controls, text="Distribution").grid(row=7, column=0, sticky="w", padx=6, pady=6)
        self.single_distribution = tk.StringVar(value=DEFAULT_SINGLE_PROP_DISTRIBUTION)
        distribution_box = ttk.Combobox(
            controls,
            textvariable=self.single_distribution,
            values=SINGLE_PROP_DISTRIBUTION_CHOICES,
            state="readonly",
            width=25,
        )
        distribution_box.grid(row=7, column=1, sticky="w", padx=6, pady=6)
        self.single_league_avg_def = self._add_labeled_slider(
            controls,
            8,
            "League Avg Def Rating",
            DEFAULT_LEAGUE_AVG_DEF_RATING,
            min_value=105.0,
            max_value=120.0,
            resolution=0.1,
        )
        self.single_defense_sensitivity = self._add_labeled_slider(
            controls,
            9,
            "Defense Sensitivity",
            DEFAULT_DEFENSE_SENSITIVITY,
            min_value=0.0,
            max_value=1.5,
            resolution=0.01,
        )
        self.single_blowout_threshold = self._add_labeled_slider(
            controls,
            10,
            "Blowout Threshold",
            DEFAULT_BLOWOUT_THRESHOLD,
            min_value=0.0,
            max_value=25.0,
            resolution=0.5,
        )
        self.single_blowout_penalty = self._add_labeled_slider(
            controls,
            11,
            "Blowout Penalty",
            DEFAULT_BLOWOUT_PENALTY,
            min_value=0.0,
            max_value=0.5,
            resolution=0.01,
        )
        self.single_defense_severity = self._add_labeled_slider(
            controls,
            12,
            "Defense Severity",
            1.0,
            min_value=0.0,
            max_value=3.0,
            resolution=0.05,
        )
        self.single_minutes_severity = self._add_labeled_slider(
            controls,
            13,
            "Minutes Penalty Severity",
            1.0,
            min_value=0.0,
            max_value=3.0,
            resolution=0.05,
        )
        self.single_sigma_severity = self._add_labeled_slider(
            controls,
            14,
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
        ).grid(row=15, column=0, sticky="w", padx=6, pady=6)

        self.single_show_plot = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Show Distribution Plot", variable=self.single_show_plot).grid(
            row=16, column=0, sticky="w", padx=6, pady=6
        )

        ttk.Button(controls, text="Run Single Prop", command=self._run_single).grid(
            row=16, column=1, sticky="w", padx=6, pady=6
        )

        self.single_output = scrolledtext.ScrolledText(tab, height=18, wrap=tk.WORD, state="disabled")
        self.single_output.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _build_parlay_tab(self, tab):
        controls = ttk.Frame(tab)
        controls.pack(fill="x", padx=8, pady=8)
        controls.columnconfigure(1, weight=1)

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
        self.parlay_corr_severity = self._add_labeled_slider(
            controls,
            8,
            "Correlation Severity",
            1.0,
            min_value=0.0,
            max_value=3.0,
            resolution=0.05,
        )
        self.parlay_vol_severity = self._add_labeled_slider(
            controls,
            9,
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
        ).grid(row=10, column=0, sticky="w", padx=6, pady=6)

        ttk.Button(controls, text="Run Parlay", command=self._run_parlay).grid(
            row=10, column=1, sticky="w", padx=6, pady=6
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
        ttk.Button(controls, text="Delete Selected Parsed Row(s)", command=self._delete_selected_manual_rows).grid(
            row=3, column=2, sticky="w", padx=6, pady=6
        )

        help_text = (
            "Paste one prop per line (pipe, csv, or tab-delimited).\n"
            "Formats:\n"
            "  1) player | stat | line | over_odds | under_odds\n"
            "  2) player | game_date | book | stat | line | over_odds | under_odds\n"
            "  3) Raw board dump (auto-extract from blocks containing player + matchup + line + stat)\n"
            "Notes: stats accept aliases (pts/ast/reb/pra). Unknown stats become snake_case. Odds are optional."
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

        table_frame = ttk.Frame(tab)
        table_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        columns = ("row", "player", "date", "book", "stat", "line", "over", "under")
        self.manual_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
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

        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.manual_table.yview)
        self.manual_table.configure(yscrollcommand=yscroll.set)
        self.manual_table.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        self.manual_output = scrolledtext.ScrolledText(tab, height=12, wrap=tk.WORD, state="disabled")
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
                line=self._parse_float_or_default(self.single_line.get(), DEFAULT_POINTS_LINE),
                rolling_window=self._parse_int_or_default(self.single_window.get(), DEFAULT_ROLLING_WINDOW),
                american_odds=self._parse_int_or_default(self.single_odds.get(), DEFAULT_AMERICAN_ODDS),
                opp_def_rating=self._parse_float_or_default(self.single_opp_def.get(), DEFAULT_OPP_DEF_RATING),
                vegas_spread=self._parse_float_or_default(self.single_spread.get(), DEFAULT_VEGAS_SPREAD),
                league_avg_def_rating=float(self.single_league_avg_def.get()),
                defense_sensitivity=float(self.single_defense_sensitivity.get()),
                blowout_threshold=float(self.single_blowout_threshold.get()),
                blowout_penalty=float(self.single_blowout_penalty.get()),
                n_games=self._parse_int_or_default(self.single_n_games.get(), 100),
                show_plot=self.single_show_plot.get(),
                distribution=self.single_distribution.get().strip().lower(),
                defense_severity=float(self.single_defense_severity.get()),
                minutes_penalty_severity=float(self.single_minutes_severity.get()),
                sigma_severity=float(self.single_sigma_severity.get()),
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
            odds_value = self._parse_float_or_default(self.parlay_odds.get(), 3.0)
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
                n_games=self._parse_int_or_default(self.parlay_n_games.get(), 100),
                n_sims=self._parse_int_or_default(self.parlay_n_sims.get(), 20000),
                correlation_severity=float(self.parlay_corr_severity.get()),
                volatility_severity=float(self.parlay_vol_severity.get()),
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
