import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

# Allow direct execution via ".../nba_model/simple_ui.py"
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

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


class SimpleModelUI:
    """Small desktop UI to run single-prop and parlay workflows."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NBA Props - Simple UI")
        self.root.geometry("900x650")

        self._build_layout()

    def _build_layout(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        single_tab = ttk.Frame(notebook)
        parlay_tab = ttk.Frame(notebook)
        notebook.add(single_tab, text="Single Prop")
        notebook.add(parlay_tab, text="Parlay (PrizePicks/Underdog)")

        self._build_single_tab(single_tab)
        self._build_parlay_tab(parlay_tab)

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
