#!/usr/bin/env python3
"""
label_trial_hits.py
====================
Adds a ``correct`` column (0 / 1) to ymaze_time_log_raw.csv by prompting
the user with a pop-up window for each (mouse, day) session.

Logic
-----
- Within each (ID, Day) session, rows are sorted by ``start_frame`` to get
  true temporal order.
- **t1 trials define the trial index** (trial 1 … N).  The number of
  checkboxes shown equals the number of t1 clips found (usually 5, but
  sometimes 3, 4, or 6 — the popup adapts automatically).
- Each checkbox = "Trial N was a correct hit".
- t2 rows are matched to the nearest preceding t1 row (same trial slot).
  If a t2 has no preceding t1 (edge case), it is placed in the first slot.
- Any row that cannot be assigned a trial slot gets ``correct = NaN``.

Usage
-----
    python label_trial_hits.py                         # reads/writes in cwd
    python label_trial_hits.py /path/to/folder         # explicit folder

Output
------
    ymaze_time_log_labeled.csv   — original CSV + new ``correct`` column
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

import pandas as pd


# ── constants ──────────────────────────────────────────────────────────────────
INPUT_FILE  = "ymaze_time_log_without_hitmisslabel.csv"
OUTPUT_FILE = "ymaze_time_log_labeled.csv"

BG     = "#16161f"
PANEL  = "#1f1f2e"
BORDER = "#2e2e45"
ACCENT = "#7c6af7"
ACC2   = "#5a4fcf"
FG     = "#f0f0f8"
DIM    = "#6a6a88"
GREEN  = "#4caf78"
RED    = "#e05555"
GOLD   = "#e8b84b"


# ── trial-slot assignment ──────────────────────────────────────────────────────

def assign_trial_slots(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given all rows for one (ID, Day), sorted by start_frame, assign each row
    a 1-based ``trial_slot`` integer.

    - t1 rows are numbered 1, 2, 3, … in the order they appear.
    - Each t2 row inherits the slot of the most recent t1 that preceded it
      (same 'trial pair').  If no t1 precedes a t2, it gets slot 1.
    """
    df = session_df.sort_values("start_frame").copy()
    slots = []
    current_t1_slot = 0

    for _, row in df.iterrows():
        if row["time_point"] == "t1":
            current_t1_slot += 1
            slots.append(current_t1_slot)
        else:
            # t2: inherit the current t1 slot (or 1 if no t1 seen yet)
            slots.append(max(current_t1_slot, 1))

    df["trial_slot"] = slots
    return df


# ── GUI ────────────────────────────────────────────────────────────────────────

class HitLabelApp(tk.Tk):
    """
    Iterates through every (ID, Day) session and shows a checkbox popup.
    The user checks which trials were correct hits, then clicks Save.
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.title("Y-Maze Trial Hit Labeller")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.df = df.copy()
        self.df["correct"] = pd.NA          # will be filled in

        # build ordered session list
        sessions = (
            df[["ID", "Day"]]
            .drop_duplicates()
            .sort_values(["ID", "Day"])
            .values.tolist()
        )
        self.sessions     = sessions
        self.session_idx  = 0
        self.hit_records  = {}   # (ID, Day, trial_slot) -> 0/1

        self._build_header()
        self._build_session_frame()
        self._build_nav()
        self._load_session()

    # ── static header ─────────────────────────────────────────────────────────

    def _build_header(self):
        # top accent bar
        tk.Frame(self, bg=ACCENT, height=3).pack(fill="x")

        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=28, pady=(18, 4))

        tk.Label(
            hdr, text="Y-Maze  ·  Trial Hit Labeller",
            font=("Helvetica", 20, "bold"),
            bg=BG, fg=FG,
        ).pack(anchor="w")

        self.subtitle = tk.Label(
            hdr, text="",
            font=("Helvetica", 13), bg=BG, fg=ACCENT,
        )
        self.subtitle.pack(anchor="w", pady=(2, 0))

        # divider
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=0, pady=(12, 0))

    # ── per-session checkbox area ──────────────────────────────────────────────

    def _build_session_frame(self):
        self.session_frame = tk.Frame(self, bg=BG)
        self.session_frame.pack(fill="x", padx=28, pady=(16, 0))

    def _build_nav(self):
        # thin divider above nav
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", pady=(16, 0))

        nav = tk.Frame(self, bg=BG)
        nav.pack(fill="x", padx=28, pady=(12, 20))

        self.progress_label = tk.Label(
            nav, text="",
            font=("Helvetica", 11), bg=BG, fg=DIM,
        )
        self.progress_label.pack(side="left")

        btn_row = tk.Frame(nav, bg=BG)
        btn_row.pack(side="right")

        self.done_btn = self._btn(btn_row, "Finish & Save", self._finish,
                                  color=GREEN, side="right")
        self.next_btn = self._btn(btn_row, "Save & Next →", self._next,
                                  color=ACCENT, side="right")
        self.prev_btn = self._btn(btn_row, "← Back", self._prev,
                                  color="#38384e", side="right")

    # ── session loader ─────────────────────────────────────────────────────────

    def _load_session(self):
        # clear old widgets
        for w in self.session_frame.winfo_children():
            w.destroy()

        mouse_id, day = self.sessions[self.session_idx]
        n_total = len(self.sessions)

        # update labels
        self.subtitle.config(
            text=f"Mouse  {mouse_id}   ·   Day {day}"
        )
        self.progress_label.config(
            text=f"Session {self.session_idx + 1} of {n_total}"
        )

        # get session data with slot assignments
        mask = (self.df["ID"] == mouse_id) & (self.df["Day"] == day)
        session_df = assign_trial_slots(self.df[mask])
        n_t1 = int((session_df["time_point"] == "t1").sum())

        # show t2 count info
        n_t2 = int((session_df["time_point"] == "t2").sum())
        has_warning = (n_t2 != n_t1)

        # instruction + trial count row
        meta_row = tk.Frame(self.session_frame, bg=BG)
        meta_row.pack(fill="x", pady=(0, 14))

        tk.Label(
            meta_row,
            text="Mark each trial as correct hit:",
            font=("Helvetica", 13), bg=BG, fg=DIM,
        ).pack(side="left")

        count_text = f"t1 × {n_t1}   t2 × {n_t2}"
        if has_warning:
            count_text += f"  ⚠ {abs(n_t1 - n_t2)} t2 missing"
        tk.Label(
            meta_row,
            text=count_text,
            font=("Helvetica", 11),
            bg=BG, fg=GOLD if has_warning else DIM,
        ).pack(side="right")

        # one row per trial slot
        self.check_vars: list[tk.BooleanVar] = []
        for slot in range(1, n_t1 + 1):
            prev = self.hit_records.get((mouse_id, day, slot), None)
            var = tk.BooleanVar(value=bool(prev) if prev is not None else False)
            self.check_vars.append(var)

            t1_rows = session_df[
                (session_df["time_point"] == "t1") &
                (session_df["trial_slot"] == slot)
            ]
            t2_rows = session_df[
                (session_df["time_point"] == "t2") &
                (session_df["trial_slot"] == slot)
            ]
            t1_info = (
                f"t1  {t1_rows.iloc[0]['start_time_s']:.1f} s"
                if len(t1_rows) else "t1  —"
            )
            t2_info = (
                f"t2  {t2_rows.iloc[0]['start_time_s']:.1f} s"
                if len(t2_rows) else "t2  —"
            )

            # card-style row
            card = tk.Frame(
                self.session_frame, bg=PANEL,
                highlightbackground=BORDER, highlightthickness=1,
            )
            card.pack(fill="x", pady=4)

            # trial label on the left
            tk.Label(
                card, text=f"Trial {slot}",
                font=("Helvetica", 14, "bold"),
                bg=PANEL, fg=FG,
                width=8, anchor="w",
            ).pack(side="left", padx=(14, 0), pady=10)

            # time stamps in the middle
            tk.Label(
                card,
                text=f"{t1_info}        {t2_info}",
                font=("Helvetica", 12),
                bg=PANEL, fg=DIM,
            ).pack(side="left", padx=20)

            # Single toggle button — the only thing to click
            is_hit = bool(prev) if prev is not None else False

            def make_toggle(v=var, c=card):
                # btn is defined right after; we capture it via a mutable list
                btn_ref = []

                def update(*_):
                    hit = v.get()
                    c.config(highlightbackground=GREEN if hit else BORDER)
                    if btn_ref:
                        btn_ref[0].config(
                            text="✓  Hit" if hit else "✗  Miss",
                            bg=GREEN  if hit else BORDER,
                            fg="black",
                            activebackground=GREEN if hit else "#3a3a5e",
                        )

                v.trace_add("write", update)
                return btn_ref, update

            btn_ref, _updater = make_toggle()

            toggle_btn = tk.Button(
                card,
                text="✓  Hit" if is_hit else "✗  Miss",
                font=("Helvetica", 12, "bold"),
                bg=GREEN  if is_hit else BORDER,
                fg="black",
                activeforeground="black",
                activebackground=GREEN if is_hit else "#3a3a5e",
                relief="flat", bd=0, cursor="hand2",
                padx=18, pady=8,
                command=lambda v=var: v.set(not v.get()),
            )
            toggle_btn.pack(side="right", padx=12, pady=8)
            btn_ref.append(toggle_btn)

            # sync card border on first load
            card.config(highlightbackground=GREEN if is_hit else BORDER)

        # quick-select row
        qs = tk.Frame(self.session_frame, bg=BG)
        qs.pack(fill="x", pady=(14, 0))
        tk.Label(qs, text="Quick select:", font=("Helvetica", 11),
                 bg=BG, fg=DIM).pack(side="left", padx=(0, 12))
        self._small_btn(qs, "All correct", lambda: self._set_all(True)).pack(side="left", padx=(0, 8))
        self._small_btn(qs, "All wrong",   lambda: self._set_all(False)).pack(side="left")

        # update nav button states
        self.prev_btn.config(state="normal" if self.session_idx > 0 else "disabled")
        is_last = (self.session_idx == len(self.sessions) - 1)
        self.next_btn.config(state="disabled" if is_last else "normal")
        self.done_btn.config(state="normal" if is_last else "disabled")

    # ── save current session's checkboxes into hit_records ────────────────────

    def _save_current(self):
        mouse_id, day = self.sessions[self.session_idx]
        for slot, var in enumerate(self.check_vars, start=1):
            self.hit_records[(mouse_id, day, slot)] = int(var.get())

    # ── navigation ────────────────────────────────────────────────────────────

    def _next(self):
        self._save_current()
        self.session_idx += 1
        self._load_session()

    def _prev(self):
        self._save_current()
        self.session_idx -= 1
        self._load_session()

    def _set_all(self, value: bool):
        for var in self.check_vars:
            var.set(value)

    # ── write correct column and save ─────────────────────────────────────────

    def _finish(self):
        self._save_current()

        # check if any sessions were skipped (no entry in hit_records)
        missing = []
        for mouse_id, day in self.sessions:
            mask = (self.df["ID"] == mouse_id) & (self.df["Day"] == day)
            session_df = assign_trial_slots(self.df[mask])
            n_t1 = int((session_df["time_point"] == "t1").sum())
            for slot in range(1, n_t1 + 1):
                if (mouse_id, day, slot) not in self.hit_records:
                    missing.append(f"{mouse_id} Day{day} Trial{slot}")

        if missing:
            ans = messagebox.askyesno(
                "Unlabelled trials",
                f"{len(missing)} trial(s) have not been labelled "
                f"(will be saved as NaN):\n\n"
                + "\n".join(missing[:20])
                + ("\n…" if len(missing) > 20 else "")
                + "\n\nSave anyway?"
            )
            if not ans:
                return

        # apply hit_records to df
        # for each row, determine its (ID, Day, trial_slot) and look up value
        result_correct = []
        for _, row in self.df.iterrows():
            mask = (
                (self.df["ID"] == row["ID"]) &
                (self.df["Day"] == row["Day"])
            )
            session_df = assign_trial_slots(self.df[mask])
            # find this row's trial_slot
            match = session_df[
                (session_df["start_frame"] == row["start_frame"]) &
                (session_df["time_point"] == row["time_point"])
            ]
            if len(match):
                slot = int(match.iloc[0]["trial_slot"])
                val = self.hit_records.get((row["ID"], row["Day"], slot), pd.NA)
            else:
                val = pd.NA
            result_correct.append(val)

        self.df["correct"] = result_correct

        # save
        out_path = os.path.join(in_dir, OUTPUT_FILE)
        self.df.to_csv(out_path, index=False)
        messagebox.showinfo(
            "Saved",
            f"Labelled CSV saved to:\n{out_path}\n\n"
            f"{self.df['correct'].notna().sum()} / {len(self.df)} rows labelled."
        )
        self.destroy()

    # ── button helpers ─────────────────────────────────────────────────────────

    def _btn(self, parent, text, cmd, big=False, color=None, side="left"):
        c = color or ACCENT
        def darken(hex_color):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            r, g, b = max(0,r-30), max(0,g-30), max(0,b-30)
            return f"#{r:02x}{g:02x}{b:02x}"
        hov = darken(c)
        b = tk.Button(
            parent, text=text, command=cmd,
            bg=c, fg="black",
            activebackground=hov, activeforeground="black",
            font=("Helvetica", 13, "bold"),
            relief="flat", bd=0, cursor="hand2",
            padx=20, pady=10,
        )
        b.bind("<Enter>", lambda e: b.config(bg=hov))
        b.bind("<Leave>", lambda e: b.config(bg=c))
        b.pack(side=side, padx=6)
        return b

    def _small_btn(self, parent, text, cmd):
        b = tk.Button(
            parent, text=text, command=cmd,
            bg=BORDER, fg="black",
            activebackground="#3a3a5e", activeforeground="black",
            font=("Helvetica", 11), relief="flat", bd=0,
            cursor="hand2", padx=16, pady=7,
        )
        b.bind("<Enter>", lambda e: b.config(bg="#3a3a5e"))
        b.bind("<Leave>", lambda e: b.config(bg=BORDER))
        return b


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    if not os.path.isdir(in_dir):
        sys.exit(f"Not a directory: {in_dir}")

    input_path  = os.path.join(in_dir, INPUT_FILE)
    output_path = os.path.join(in_dir, OUTPUT_FILE)

    if not os.path.exists(input_path):
        sys.exit(f"Input file not found: {input_path}")

    raw_df = pd.read_csv(input_path)

    required = {"ID", "Day", "time_point", "start_frame", "start_time_s"}
    missing_cols = required - set(raw_df.columns)
    if missing_cols:
        sys.exit(f"Missing columns in CSV: {sorted(missing_cols)}")

    raw_df["Day"]         = pd.to_numeric(raw_df["Day"],         errors="coerce")
    raw_df["start_frame"] = pd.to_numeric(raw_df["start_frame"], errors="coerce")

    all_sessions = (
        raw_df[["ID", "Day"]]
        .drop_duplicates()
        .sort_values(["ID", "Day"])
        .values.tolist()
    )

    # ── Resume logic ─────────────────────────────────────────────────────────
    if os.path.exists(output_path):
        labeled_df = pd.read_csv(output_path)
        labeled_df["Day"]         = pd.to_numeric(labeled_df["Day"],         errors="coerce")
        labeled_df["start_frame"] = pd.to_numeric(labeled_df["start_frame"], errors="coerce")

        def session_is_complete(mouse_id, day):
            # Done = present in labeled file with no NaN correct values
            mask = (labeled_df["ID"] == mouse_id) & (labeled_df["Day"] == day)
            rows = labeled_df[mask]
            return len(rows) > 0 and rows["correct"].notna().all()

        todo_sessions = [
            [mid, day] for mid, day in all_sessions
            if not session_is_complete(mid, day)
        ]

        n_done = len(all_sessions) - len(todo_sessions)
        print(f"Labeled file found: {output_path}")
        print(f"  {n_done} / {len(all_sessions)} sessions already complete.")

        if not todo_sessions:
            print("All sessions already labeled — nothing to do.")
            sys.exit(0)

        print(f"  {len(todo_sessions)} session(s) still need labeling:")
        for mid, day in todo_sessions:
            print(f"    {mid}  Day {int(day)}")

        todo_mask = raw_df.apply(lambda r: [r["ID"], r["Day"]] in todo_sessions, axis=1)
        df = raw_df[todo_mask].copy()
        df["correct"] = pd.NA

    else:
        print(f"No labeled file found — starting fresh ({len(all_sessions)} sessions).")
        labeled_df    = None
        todo_sessions = all_sessions
        df = raw_df.copy()
        df["correct"] = pd.NA

    print(f"Opening GUI for {len(todo_sessions)} session(s), {len(df)} rows.")

    # Capture for use inside patched _finish
    _out_path = output_path
    _existing = labeled_df
    _raw      = raw_df

    def _patched_finish(self):
        self._save_current()

        missing = []
        for mouse_id, day in self.sessions:
            mask = (self.df["ID"] == mouse_id) & (self.df["Day"] == day)
            session_df = assign_trial_slots(self.df[mask])
            n_t1 = int((session_df["time_point"] == "t1").sum())
            for slot in range(1, n_t1 + 1):
                if (mouse_id, day, slot) not in self.hit_records:
                    missing.append(f"{mouse_id} Day{day} Trial{slot}")

        if missing:
            ans = messagebox.askyesno(
                "Unlabelled trials",
                f"{len(missing)} trial(s) not yet labelled (will be NaN):\n\n"
                + "\n".join(missing[:20])
                + ("\n\u2026" if len(missing) > 20 else "")
                + "\n\nSave anyway?"
            )
            if not ans:
                return

        result_correct = []
        for _, row in self.df.iterrows():
            mask = (self.df["ID"] == row["ID"]) & (self.df["Day"] == row["Day"])
            session_df = assign_trial_slots(self.df[mask])
            match = session_df[
                (session_df["start_frame"] == row["start_frame"]) &
                (session_df["time_point"]  == row["time_point"])
            ]
            if len(match):
                slot = int(match.iloc[0]["trial_slot"])
                val  = self.hit_records.get((row["ID"], row["Day"], slot), pd.NA)
            else:
                val = pd.NA
            result_correct.append(val)

        self.df["correct"] = result_correct

        # Merge new labels into existing file (or create fresh)
        merge_keys = ["ID", "Day", "start_frame", "time_point"]
        new_labels = self.df[merge_keys + ["correct"]].copy()

        if _existing is not None:
            base = _existing.copy()
            session_pairs = self.df[["ID", "Day"]].drop_duplicates()
            for _, sp in session_pairs.iterrows():
                drop_mask = (base["ID"] == sp["ID"]) & (base["Day"] == sp["Day"])
                base = base[~drop_mask]
            new_full = _raw.merge(new_labels, on=merge_keys, how="inner")
            out_df = pd.concat([base, new_full], ignore_index=True)
        else:
            out_df = _raw.merge(new_labels, on=merge_keys, how="left")

        out_df = out_df.sort_values(["ID", "Day", "start_frame"]).reset_index(drop=True)
        out_df.to_csv(_out_path, index=False)

        n_labeled = out_df["correct"].notna().sum()
        messagebox.showinfo(
            "Saved",
            f"Updated:\n{_out_path}\n\n"
            f"{n_labeled} / {len(out_df)} rows labeled across all sessions."
        )
        self.destroy()

    HitLabelApp._finish = _patched_finish

    app = HitLabelApp(df)
    app.mainloop()