#!/usr/bin/env python3
"""
step1b_fix_clips.py
===================
Load a *_clips.csv, highlight suspicious rows, and let you fix individual
clips without re-annotating everything.

Suspicious = a clip's stop_frame is within THRESHOLD frames of any adjacent
clip's start_frame or stop_frame (catches the "stop set at wrong boundary" bug).

Workflow:
  1. Open a *_clips.csv  (and optionally the source video)
  2. Suspicious rows are highlighted in red
  3. Click a row → video jumps to that clip's region so you can verify
  4. Edit start/stop/label in-place → "Fix Selected" re-extracts just that clip
     and updates the CSV

Dependencies:
    pip install opencv-python pillow pandas
"""

import os
import csv
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import pandas as pd
from PIL import Image, ImageTk

THRESHOLD = 100   # frames — flag if stop is within this distance of a neighbour boundary


# ── helpers ────────────────────────────────────────────────────────────────────

def load_csv(path):
    df = pd.read_csv(path)
    for col in ("start_frame", "stop_frame", "n_frames"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def flag_suspicious(df):
    """Return a boolean Series: True where a T2 clip's stop_frame is within
    THRESHOLD frames of the next clip's start_frame."""
    flags  = pd.Series(False, index=df.index)
    labels = df["time_point"].astype(str).str.strip().str.lower().tolist()
    stops  = pd.to_numeric(df["stop_frame"],  errors="coerce").tolist()
    starts = pd.to_numeric(df["start_frame"], errors="coerce").tolist()

    for i in range(len(df) - 1):
        if labels[i] != "t2":
            continue
        t2_stop    = stops[i]
        next_start = starts[i + 1]
        if pd.isna(t2_stop) or pd.isna(next_start):
            continue
        if 0 < abs(t2_stop - next_start) <= THRESHOLD:
            flags.iloc[i] = True
    return flags


# ── main app ───────────────────────────────────────────────────────────────────

class FixClipsApp:
    MAX_DISPLAY_W = 760
    MAX_DISPLAY_H = 420

    def __init__(self, root):
        self.root = root
        self.root.title("Clip Fix Tool")
        self.root.minsize(900, 750)

        self.csv_path  = None
        self.df        = None
        self.flags     = None
        self.cap       = None
        self.fps       = 30.0
        self.frame_w   = 0
        self.frame_h   = 0
        self.total_frames = 0
        self._photo    = None
        self.current_frame_idx = 0
        self.pending_start = None

        self._build_ui()
        self._bind_keys()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # top bar
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Open CSV",   command=self.open_csv).pack(side=tk.LEFT)
        ttk.Button(top, text="Open Video", command=self.open_video_dialog).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Save CSV",   command=self.save_csv).pack(side=tk.RIGHT)
        self.info_label = ttk.Label(top, text="No file loaded")
        self.info_label.pack(side=tk.LEFT, padx=12)

        # paned: table top, video bottom
        paned = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── table panel ──────────────────────────────────────────────────────
        tbl_frame = ttk.Frame(paned, padding=4)
        paned.add(tbl_frame, minsize=180)

        # use grid exclusively in tbl_frame to avoid pack/grid conflict
        ttk.Label(tbl_frame,
            text=f"Red = stop_frame within ±{THRESHOLD} frames of an adjacent clip's boundary",
            foreground="red").grid(row=0, column=0, columnspan=2, sticky="w")

        cols = ("clip_index", "ID", "Day", "time_point",
                "start_frame", "stop_frame", "n_frames", "duration_s", "clip_filename")
        self.tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=10,
                                 selectmode="browse")
        heads = {"clip_index": "#", "ID": "ID", "Day": "Day", "time_point": "t1/t2",
                 "start_frame": "Start", "stop_frame": "Stop",
                 "n_frames": "Frames", "duration_s": "Dur(s)", "clip_filename": "Filename"}
        widths = {"clip_index": 35, "ID": 60, "Day": 40, "time_point": 45,
                  "start_frame": 70, "stop_frame": 70, "n_frames": 65,
                  "duration_s": 65, "clip_filename": 280}
        for c in cols:
            self.tree.heading(c, text=heads[c])
            self.tree.column(c, width=widths[c], anchor=tk.CENTER)
        self.tree.tag_configure("suspect", background="#ffcccc")
        self.tree.tag_configure("normal",  background="")

        vsb = ttk.Scrollbar(tbl_frame, orient=tk.VERTICAL,   command=self.tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")
        hsb.grid(row=2, column=0, sticky="ew")
        tbl_frame.rowconfigure(1, weight=1)
        tbl_frame.columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", self._on_row_select)

        # ── edit + fix bar ────────────────────────────────────────────────────
        edit_frame = ttk.LabelFrame(tbl_frame, text="Edit selected row", padding=6)
        edit_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=4)

        for col_name, label_text in (("start_frame", "Start frame:"),
                                      ("stop_frame",  "Stop frame:"),
                                      ("time_point",  "Label (t1/t2):")):
            ttk.Label(edit_frame, text=label_text).pack(side=tk.LEFT)
            ent = ttk.Entry(edit_frame, width=10)
            ent.pack(side=tk.LEFT, padx=(0, 12))
            setattr(self, f"_ent_{col_name}", ent)

        ttk.Button(edit_frame, text="Set Start = current frame",
                   command=self._set_start_from_video).pack(side=tk.LEFT, padx=4)
        ttk.Button(edit_frame, text="Set Stop = current frame",
                   command=self._set_stop_from_video).pack(side=tk.LEFT, padx=4)
        ttk.Button(edit_frame, text="Fix Selected (re-extract clip)",
                   command=self.fix_selected).pack(side=tk.RIGHT)

        # ── video panel ──────────────────────────────────────────────────────
        vid_frame = ttk.Frame(paned, padding=4)
        paned.add(vid_frame, minsize=200)

        self.canvas = tk.Canvas(vid_frame, bg="black",
                                width=self.MAX_DISPLAY_W, height=self.MAX_DISPLAY_H)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        ctrl = ttk.Frame(vid_frame, padding=4)
        ctrl.pack(fill=tk.X)
        ttk.Button(ctrl, text="<< -10",  command=lambda: self.step(-10)).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="< -1",    command=lambda: self.step(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="+1 >",    command=lambda: self.step(1)).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="+10 >>",  command=lambda: self.step(10)).pack(side=tk.LEFT, padx=2)

        self.slider = ttk.Scale(ctrl, from_=0, to=0, orient=tk.HORIZONTAL,
                                command=self._on_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.frame_entry = ttk.Entry(ctrl, width=8)
        self.frame_entry.pack(side=tk.LEFT)
        self.frame_entry.bind("<Return>", self._on_frame_entry)
        ttk.Button(ctrl, text="Go", command=self._on_frame_entry).pack(side=tk.LEFT, padx=2)

        # status
        self.status = ttk.Label(self.root, text="Ready.", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self):
        self.root.bind("<Left>",       lambda e: self.step(-1))
        self.root.bind("<Right>",      lambda e: self.step(1))
        self.root.bind("<Shift-Left>", lambda e: self.step(-10))
        self.root.bind("<Shift-Right>",lambda e: self.step(10))

    # ── CSV ───────────────────────────────────────────────────────────────────

    def open_csv(self):
        path = filedialog.askopenfilename(
            title="Select *_clips.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        self.csv_path = path
        self.df = load_csv(path)
        self.flags = flag_suspicious(self.df)
        self._populate_tree()
        n_sus = self.flags.sum()
        self.info_label.config(text=f"{os.path.basename(path)}  |  {n_sus} suspicious row(s)")
        self.set_status(f"Loaded {len(self.df)} clips, {n_sus} flagged.")

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        cols = ("clip_index", "ID", "Day", "time_point",
                "start_frame", "stop_frame", "n_frames", "duration_s", "clip_filename")
        for i, row in self.df.iterrows():
            tag = "suspect" if self.flags.iloc[i] else "normal"
            vals = tuple(row.get(c, "") for c in cols)
            self.tree.insert("", tk.END, iid=str(i), values=vals, tags=(tag,))

    def save_csv(self):
        if self.df is None:
            return
        self.df.to_csv(self.csv_path, index=False)
        self.set_status(f"Saved → {self.csv_path}")

    # ── row selection → populate edit fields + jump video ────────────────────

    def _on_row_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        i = int(sel[0])
        row = self.df.iloc[i]
        self._ent_start_frame.delete(0, tk.END)
        self._ent_start_frame.insert(0, str(int(row.get("start_frame", ""))))
        self._ent_stop_frame.delete(0, tk.END)
        self._ent_stop_frame.insert(0, str(int(row.get("stop_frame", ""))))
        self._ent_time_point.delete(0, tk.END)
        self._ent_time_point.insert(0, str(row.get("time_point", "")))

        # jump video to 50 frames before the clip start so user has context
        if self.cap is not None:
            start = int(row.get("start_frame", 0))
            self.show_frame(max(0, start - 50))

    def _set_start_from_video(self):
        self._ent_start_frame.delete(0, tk.END)
        self._ent_start_frame.insert(0, str(self.current_frame_idx))

    def _set_stop_from_video(self):
        self._ent_stop_frame.delete(0, tk.END)
        self._ent_stop_frame.insert(0, str(self.current_frame_idx))

    # ── fix ───────────────────────────────────────────────────────────────────

    def fix_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Nothing selected", "Click a row first.")
            return
        i = int(sel[0])

        try:
            new_start = int(self._ent_start_frame.get())
            new_stop  = int(self._ent_stop_frame.get())
        except ValueError:
            messagebox.showerror("Bad value", "Start/stop must be integers.")
            return
        new_label = self._ent_time_point.get().strip() or str(self.df.at[i, "time_point"])

        if new_stop < new_start:
            messagebox.showerror("Bad range", "Stop must be ≥ start.")
            return

        # update dataframe
        old_clip_name = str(self.df.at[i, "clip_filename"])
        old_clip_dir  = os.path.dirname(self.csv_path)

        fps = float(self.df.at[i, "fps"]) if "fps" in self.df.columns else self.fps
        n = new_stop - new_start + 1
        dur = (new_stop - new_start) / fps if fps else 0

        # build new clip filename (same base, updated frames)
        base_vid = str(self.df.at[i, "source_video"]) if "source_video" in self.df.columns else ""
        base = os.path.splitext(base_vid)[0] if base_vid else old_clip_name.rsplit("_clip", 1)[0]
        clip_idx = int(self.df.at[i, "clip_index"])
        new_clip_name = f"{base}_clip{clip_idx:03d}_{new_label}_f{new_start}-{new_stop}.mp4"

        self.df.at[i, "start_frame"] = new_start
        self.df.at[i, "stop_frame"]  = new_stop
        self.df.at[i, "n_frames"]    = n
        self.df.at[i, "duration_s"]  = f"{dur:.4f}"
        self.df.at[i, "time_point"]  = new_label
        self.df.at[i, "clip_filename"] = new_clip_name
        if "start_time_s" in self.df.columns:
            self.df.at[i, "start_time_s"] = f"{new_start / fps:.4f}"
        if "stop_time_s" in self.df.columns:
            self.df.at[i, "stop_time_s"]  = f"{new_stop  / fps:.4f}"

        # re-extract clip if video is loaded
        if self.cap is not None:
            new_clip_path = os.path.join(old_clip_dir, new_clip_name)
            self.set_status(f"Extracting {new_clip_name} …")
            threading.Thread(
                target=self._extract_clip,
                args=(new_start, new_stop, new_clip_path, old_clip_dir, old_clip_name),
                daemon=True,
            ).start()
        else:
            self.set_status("CSV updated (no video loaded — clip file not re-extracted).")

        # refresh flags and tree
        self.flags = flag_suspicious(self.df)
        self._populate_tree()
        self.tree.selection_set(str(i))
        self.save_csv()

    def _extract_clip(self, start, stop, new_path, clip_dir, old_name):
        cap = cv2.VideoCapture(self.cap.get(cv2.CAP_PROP_POS_AVI_RATIO) if False else self._video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(new_path, fourcc, self.fps, (self.frame_w, self.frame_h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(start, stop + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
        cap.release()

        # remove old clip file if name changed and file exists
        old_path = os.path.join(clip_dir, old_name)
        if old_name != os.path.basename(new_path) and os.path.isfile(old_path):
            os.remove(old_path)

        self.root.after(0, lambda: self.set_status(f"Done → {os.path.basename(new_path)}"))

    # ── video ─────────────────────────────────────────────────────────────────

    def open_video_dialog(self):
        path = filedialog.askopenfilename(
            title="Select source video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All files", "*.*")])
        if path:
            self._load_video(path)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open:\n{path}")
            return
        if self.cap:
            self.cap.release()
        self.cap = cap
        self._video_path = path
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.slider.configure(to=max(self.total_frames - 1, 0))
        self.show_frame(0)
        self.set_status(f"Video loaded: {os.path.basename(path)}")

    def show_frame(self, idx):
        if self.cap is None:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_frame_idx = idx

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = min(self.MAX_DISPLAY_W / w, self.MAX_DISPLAY_H / h, 1.0)
        frame_resized = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
        img = Image.fromarray(frame_resized)
        self._photo = ImageTk.PhotoImage(image=img)
        self.canvas.config(width=int(w * scale), height=int(h * scale))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        t = idx / self.fps
        self.canvas.create_text(8, 8, anchor=tk.NW,
            text=f"Frame {idx}/{self.total_frames - 1}  |  {t:.3f}s",
            fill="yellow", font=("Helvetica", 11, "bold"))

        self.slider.set(idx)
        self.frame_entry.delete(0, tk.END)
        self.frame_entry.insert(0, str(idx))

    def step(self, delta):
        if self.cap:
            self.show_frame(self.current_frame_idx + delta)

    def _on_slider(self, value):
        idx = int(float(value))
        if idx != self.current_frame_idx:
            self.show_frame(idx)

    def _on_frame_entry(self, _event=None):
        try:
            self.show_frame(int(self.frame_entry.get()))
        except ValueError:
            pass

    def set_status(self, text):
        self.status.config(text=text)

    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = FixClipsApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
