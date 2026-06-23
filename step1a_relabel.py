#!/usr/bin/env python3
"""
relabel_mice.py
===============
Fix mislabeled mouse IDs and/or group assignments in:
  - ymaze_time_log_raw.csv
  - ymaze_time_log_labeled.csv       (if present)
  - any *_clips.csv files            (if present)

Each rule is a row in the table:
    Old ID  →  New ID  |  New Group (optional)

Rules are applied to ALL matching CSV files in the folder so everything
stays consistent.  A timestamped backup of each file is saved before any
changes are written.

Usage:
    python relabel_mice.py                   # uses current directory
    python relabel_mice.py /path/to/folder
"""

import os
import sys
import glob
import shutil
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import pandas as pd


# ── constants ──────────────────────────────────────────────────────────────────
GROUPS = ["SNr-DTA", "Ctrl", ""]   # "" = keep current group; user can type a custom label too

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


# ── CSV helpers ────────────────────────────────────────────────────────────────

def find_csvs(folder_or_file: str) -> list[str]:
    """Return CSV files to process.
    - If a single file path is given, return just that file.
    - If a folder is given, scan for ymaze/clips CSVs.
    """
    if os.path.isfile(folder_or_file):
        return [folder_or_file]

    folder = folder_or_file
    candidates = (
        glob.glob(os.path.join(folder, "ymaze_time_log_raw*.csv")) +
        glob.glob(os.path.join(folder, "ymaze_time_log_labeled*.csv")) +
        glob.glob(os.path.join(folder, "*_clips.csv"))
    )
    seen, out = set(), []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)


def backup(path: str) -> str:
    """Copy file to <name>_backup_YYYYMMDD_HHMMSS.csv and return backup path."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%SS")
    base = os.path.splitext(path)[0]
    dest = f"{base}_backup_{ts}.csv"
    shutil.copy2(path, dest)
    return dest


def apply_rules(df: pd.DataFrame, rules: list[dict]) -> tuple[pd.DataFrame, int]:
    """
    Apply relabeling rules to df.
    Each rule: {"old_id": str, "new_id": str, "new_group": str or ""}
    Returns (updated_df, n_rows_changed).
    """
    df = df.copy()
    changed = 0
    for rule in rules:
        old_id    = rule["old_id"].strip()
        new_id    = rule["new_id"].strip()
        new_group = rule["new_group"].strip()

        if not old_id or not new_id:
            continue

        mask = df["ID"].astype(str) == old_id
        n = int(mask.sum())
        if n == 0:
            continue

        df.loc[mask, "ID"] = new_id
        if new_group:
            df.loc[mask, "Group"] = new_group
        changed += n

    return df, changed


# ── GUI ────────────────────────────────────────────────────────────────────────

class RelabelApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mouse Relabeler")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(680, 480)

        self.folder_var = tk.StringVar()
        self.rules: list[dict] = []   # list of {"old_id", "new_id", "new_group"}

        self._build_ui()
        self._add_rule_row()   # start with one blank row

        # auto-fill folder from CLI arg
        if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
            self.folder_var.set(sys.argv[1])
            self._scan_folder()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # accent bar
        tk.Frame(self, bg=ACCENT, height=3).pack(fill="x")

        # title
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(14, 4))
        tk.Label(hdr, text="Mouse Relabeler",
                 font=("Helvetica", 18, "bold"), bg=BG, fg=FG).pack(anchor="w")
        tk.Label(hdr, text="Fix mislabeled IDs and/or group assignments across all CSV files.",
                 font=("Helvetica", 10), bg=BG, fg=DIM).pack(anchor="w")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", pady=(10, 0))

        # folder picker
        folder_row = tk.Frame(self, bg=BG)
        folder_row.pack(fill="x", padx=24, pady=(12, 4))
        tk.Label(folder_row, text="Folder or File:", font=("Helvetica", 10),
                 bg=BG, fg=FG).pack(side="left", padx=(0, 8))
        tk.Entry(folder_row, textvariable=self.folder_var,
                 bg="#13131f", fg=FG, insertbackground=FG,
                 relief="flat", bd=4, font=("Courier", 10), width=44
                 ).pack(side="left")
        self._btn(folder_row, "Folder…", self._browse).pack(side="left", padx=(8,2))
        self._btn(folder_row, "File…", self._browse_file).pack(side="left", padx=(0,8))
        self._btn(folder_row, "Scan", self._scan_folder).pack(side="left")

        # detected files area
        self.files_label = tk.Label(self, text="",
                                    font=("Helvetica", 8), bg=BG, fg=DIM,
                                    justify="left", anchor="w")
        self.files_label.pack(fill="x", padx=24)

        # detected mice area
        self.mice_label = tk.Label(self, text="",
                                   font=("Courier", 9), bg=BG, fg=GOLD,
                                   justify="left", anchor="w")
        self.mice_label.pack(fill="x", padx=24, pady=(2, 8))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # rules section
        tk.Label(self, text="Relabeling rules",
                 font=("Helvetica", 11, "bold"), bg=BG, fg=FG
                 ).pack(anchor="w", padx=24, pady=(10, 2))
        tk.Label(self,
                 text="Leave 'New Group' blank to keep the existing group. Select from the dropdown or type a custom label.",
                 font=("Helvetica", 9), bg=BG, fg=DIM
                 ).pack(anchor="w", padx=24)

        # column headers
        hrow = tk.Frame(self, bg=BG)
        hrow.pack(fill="x", padx=24, pady=(6, 0))
        for txt, w in [("Old ID", 110), ("→  New ID", 110), ("New Group", 130), ("", 30)]:
            tk.Label(hrow, text=txt, font=("Helvetica", 9, "bold"),
                     bg=BG, fg=DIM, width=w//7, anchor="w").pack(side="left", padx=4)

        # scrollable rule rows
        self.rules_frame = tk.Frame(self, bg=BG)
        self.rules_frame.pack(fill="x", padx=24)
        self.rule_widgets: list[dict] = []   # each: {frame, old_var, new_var, grp_var}

        # add-rule button
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(anchor="w", padx=24, pady=(6, 0))
        self._btn(btn_row, "+ Add rule", self._add_rule_row).pack(side="left")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", pady=(12, 0))

        # action buttons
        act = tk.Frame(self, bg=BG)
        act.pack(fill="x", padx=24, pady=(10, 18))
        self._btn(act, "Preview changes", self._preview,
                  color="#38384e").pack(side="left", padx=(0, 8))
        self._btn(act, "✓  Apply & Save", self._apply,
                  color=GREEN).pack(side="left")
        self.status_var = tk.StringVar(value="")
        tk.Label(act, textvariable=self.status_var,
                 font=("Helvetica", 9), bg=BG, fg=DIM
                 ).pack(side="right")

    # ── rule row ───────────────────────────────────────────────────────────────

    def _add_rule_row(self, old_id="", new_id="", new_group=""):
        row = tk.Frame(self.rules_frame, bg=BG)
        row.pack(fill="x", pady=3)

        old_var = tk.StringVar(value=old_id)
        new_var = tk.StringVar(value=new_id)
        grp_var = tk.StringVar(value=new_group)

        old_entry = tk.Entry(row, textvariable=old_var,
                             bg=PANEL, fg=FG, insertbackground=FG,
                             relief="flat", bd=3, font=("Courier", 11), width=12)
        old_entry.pack(side="left", padx=4)

        tk.Label(row, text="→", bg=BG, fg=DIM,
                 font=("Helvetica", 11)).pack(side="left")

        new_entry = tk.Entry(row, textvariable=new_var,
                             bg=PANEL, fg=FG, insertbackground=FG,
                             relief="flat", bd=3, font=("Courier", 11), width=12)
        new_entry.pack(side="left", padx=4)

        grp_menu = ttk.Combobox(row, textvariable=grp_var,
                                values=GROUPS, width=14, state="normal")
        grp_menu.pack(side="left", padx=4)

        del_btn = self._btn(row, "✕", lambda r=row: self._delete_rule_row(r),
                            color=RED)
        del_btn.pack(side="left", padx=4)

        self.rule_widgets.append({
            "frame": row, "old_var": old_var,
            "new_var": new_var, "grp_var": grp_var
        })

    def _delete_rule_row(self, frame):
        self.rule_widgets = [w for w in self.rule_widgets if w["frame"] is not frame]
        frame.destroy()

    def _collect_rules(self) -> list[dict]:
        rules = []
        for w in self.rule_widgets:
            old = w["old_var"].get().strip()
            new = w["new_var"].get().strip()
            grp = w["grp_var"].get().strip()
            if old:
                rules.append({"old_id": old, "new_id": new, "new_group": grp})
        return rules

    # ── folder scanning ────────────────────────────────────────────────────────

    def _browse(self):
        folder = filedialog.askdirectory(title="Select folder containing CSV files")
        if folder:
            self.folder_var.set(folder)
            self._scan_folder()

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select a CSV file to relabel",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.folder_var.set(path)
            self._scan_folder()

    def _scan_folder(self):
        folder = self.folder_var.get().strip()
        if not folder:
            self.files_label.config(text="No path selected.")
            return
        if not os.path.isfile(folder) and not os.path.isdir(folder):
            self.files_label.config(text="Invalid folder or file path.")
            return

        csvs = find_csvs(folder)
        if not csvs:
            self.files_label.config(text="No matching CSV files found.")
            self.mice_label.config(text="")
            return

        self.files_label.config(
            text=f"Found {len(csvs)} file(s):  " +
                 "  |  ".join(os.path.basename(f) for f in csvs)
        )

        # collect all unique IDs and their groups across files
        id_groups: dict[str, str] = {}
        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path, usecols=["ID", "Group"])
                for _, row in df.drop_duplicates().iterrows():
                    mid = str(row["ID"])
                    grp = str(row["Group"]) if "Group" in df.columns else ""
                    id_groups[mid] = grp
            except Exception:
                pass

        if id_groups:
            lines = [f"{mid} ({grp})" for mid, grp in sorted(id_groups.items())]
            self.mice_label.config(text="Detected mice:  " + "   ".join(lines))
        else:
            self.mice_label.config(text="No ID column found in CSV files.")

    # ── preview & apply ────────────────────────────────────────────────────────

    def _preview(self):
        folder = self.folder_var.get().strip()
        rules  = self._collect_rules()

        if not folder or (not os.path.isdir(folder) and not os.path.isfile(folder)):
            messagebox.showwarning("No path", "Please select a valid folder or file.")
            return
        if not any(r["old_id"] and r["new_id"] for r in rules):
            messagebox.showwarning("No rules", "Add at least one relabeling rule.")
            return

        csvs = find_csvs(folder)
        lines = ["Preview of changes:\n"]
        total_changed = 0

        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                lines.append(f"  ! Could not read {os.path.basename(csv_path)}: {e}")
                continue

            if "ID" not in df.columns:
                continue

            _, changed = apply_rules(df, rules)
            if changed:
                total_changed += changed
                lines.append(f"  {os.path.basename(csv_path)}: {changed} row(s) affected")
            else:
                lines.append(f"  {os.path.basename(csv_path)}: no matches")

        if total_changed == 0:
            lines.append("\nNo rows match the given rules — check your Old IDs.")
        else:
            lines.append(f"\nTotal: {total_changed} row(s) will be updated.")
            lines.append("A timestamped backup will be saved for each file.")

        messagebox.showinfo("Preview", "\n".join(lines))

    def _apply(self):
        folder = self.folder_var.get().strip()
        rules  = self._collect_rules()

        if not folder or (not os.path.isdir(folder) and not os.path.isfile(folder)):
            messagebox.showwarning("No path", "Please select a valid folder or file.")
            return
        if not any(r["old_id"] and r["new_id"] for r in rules):
            messagebox.showwarning("No rules", "Add at least one relabeling rule.")
            return

        csvs = find_csvs(folder)
        if not csvs:
            messagebox.showwarning("No files", "No matching CSV files found.")
            return

        # final confirm
        rule_lines = "\n".join(
            f"  {r['old_id']} → {r['new_id']}"
            + (f"  [Group: {r['new_group']}]" if r["new_group"] else "")
            for r in rules if r["old_id"] and r["new_id"]
        )
        if not messagebox.askyesno(
            "Confirm",
            f"Apply these rules to {len(csvs)} file(s)?\n\n{rule_lines}\n\n"
            "Backups will be created before saving."
        ):
            return

        summary = []
        total   = 0
        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                summary.append(f"  ! {os.path.basename(csv_path)}: read error — {e}")
                continue

            if "ID" not in df.columns:
                summary.append(f"  {os.path.basename(csv_path)}: skipped (no ID column)")
                continue

            new_df, changed = apply_rules(df, rules)
            if changed == 0:
                summary.append(f"  {os.path.basename(csv_path)}: no matches")
                continue

            bak = backup(csv_path)
            new_df.to_csv(csv_path, index=False)
            total += changed
            summary.append(
                f"  {os.path.basename(csv_path)}: {changed} row(s) updated  "
                f"(backup: {os.path.basename(bak)})"
            )

        summary.insert(0, f"Done — {total} row(s) updated across {len(csvs)} file(s).\n")
        messagebox.showinfo("Applied", "\n".join(summary))
        self.status_var.set(f"✓ {total} rows updated.")
        self._scan_folder()   # refresh detected mice display

    # ── button helper ──────────────────────────────────────────────────────────

    def _btn(self, parent, text, cmd, color=None):
        c   = color or ACCENT
        hov = ACC2
        b   = tk.Button(
            parent, text=text, command=cmd,
            bg=c, fg="black",
            activebackground=hov, activeforeground="black",
            font=("Helvetica", 10, "bold"),
            relief="flat", bd=0, cursor="hand2",
            padx=12, pady=6,
        )
        b.bind("<Enter>", lambda e: b.config(bg=hov))
        b.bind("<Leave>", lambda e: b.config(bg=c))
        return b


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = RelabelApp()
    app.mainloop()