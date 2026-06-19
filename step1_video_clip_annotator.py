#!/usr/bin/env python3
"""
Video Clip Annotator
=====================
A GUI tool for annotating a video with multiple (start, stop) frame pairs and
exporting:
  1. A CSV file recording each clip's start/stop frame + timestamp info.
  2. The corresponding video clips cut from the source video.

Dependencies:
    pip install opencv-python pillow

Usage:
    python step1_video_clip_annotator.py
    (or pass a video path directly)
    python step1_video_clip_annotator.py /path/to/video.mp4

Controls:
    - Open Video ............ load a video file
    - Slider / arrow keys ... scrub through frames
    - "Set Start" (s) ....... mark current frame as a clip start
    - "Set Stop"  (e) ....... mark current frame as a clip stop, adds the pair
    - Add Pair .............. commit the current start/stop pair to the list
    - Delete ................ remove selected pair(s) from the list
    - Export ................ write CSV + extract all clips
    - Left / Right .......... step 1 frame
    - Shift+Left / Right .... step 10 frames
"""

import os
import sys
import csv
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk


class VideoClipAnnotator:
    # Maximum dimension (px) for the on-screen preview; the source video is
    # never modified, this only affects display scaling.
    MAX_DISPLAY_W = 800
    MAX_DISPLAY_H = 450

    def __init__(self, root, initial_video=None):
        self.root = root
        self.root.title("Video Clip Annotator")
        self.root.minsize(820, 760)

        # --- video state ---------------------------------------------------
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.fps = 30.0
        self.frame_w = 0
        self.frame_h = 0
        self.current_frame_idx = 0
        self._photo = None  # keep a reference so Tk doesn't GC the image

        # pending pair being built
        self.pending_start = None

        # list of dicts: {"start": int, "stop": int}
        self.pairs = []

        self._build_ui()
        self._bind_keys()

        if initial_video:
            self.load_video(initial_video)

    # ----------------------------------------------------------------- UI --
    def _build_ui(self):
        # Top bar -----------------------------------------------------------
        topbar = ttk.Frame(self.root, padding=6)
        topbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(topbar, text="Open Video", command=self.open_video_dialog).pack(side=tk.LEFT)
        ttk.Button(topbar, text="★ Export CSV + Clips", command=self.export).pack(side=tk.RIGHT)
        self.info_label = ttk.Label(topbar, text="No video loaded")
        self.info_label.pack(side=tk.LEFT, padx=12)

        # Video display -----------------------------------------------------
        self.canvas = tk.Canvas(self.root, bg="black",
                                width=self.MAX_DISPLAY_W, height=self.MAX_DISPLAY_H)
        self.canvas.pack(side=tk.TOP, padx=6, pady=6)

        # Scrub slider ------------------------------------------------------
        slider_frame = ttk.Frame(self.root, padding=(6, 0))
        slider_frame.pack(side=tk.TOP, fill=tk.X)

        self.frame_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL,
                                command=self._on_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.frame_entry = ttk.Entry(slider_frame, width=10, justify=tk.RIGHT)
        self.frame_entry.pack(side=tk.LEFT, padx=6)
        self.frame_entry.bind("<Return>", self._on_frame_entry)
        ttk.Button(slider_frame, text="Go", command=self._on_frame_entry).pack(side=tk.LEFT)

        # Step buttons ------------------------------------------------------
        step_frame = ttk.Frame(self.root, padding=6)
        step_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(step_frame, text="<< -10", command=lambda: self.step(-10)).pack(side=tk.LEFT)
        ttk.Button(step_frame, text="< -1", command=lambda: self.step(-1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(step_frame, text="+1 >", command=lambda: self.step(1)).pack(side=tk.LEFT)
        ttk.Button(step_frame, text="+10 >>", command=lambda: self.step(10)).pack(side=tk.LEFT, padx=4)

        # Mark controls -----------------------------------------------------
        mark_frame = ttk.Frame(self.root, padding=6)
        mark_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(mark_frame, text="Set Start (s)", command=self.set_start).pack(side=tk.LEFT)
        ttk.Button(mark_frame, text="Set Stop (e)", command=self.set_stop).pack(side=tk.LEFT, padx=6)
        self.pending_label = ttk.Label(mark_frame, text="Pending start: --")
        self.pending_label.pack(side=tk.LEFT, padx=12)

        # Pair list + controls ---------------------------------------------
        list_frame = ttk.Frame(self.root, padding=6)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        cols = ("idx", "label", "start", "stop", "n_frames", "start_t", "stop_t", "dur_s")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=6)
        headings = {
            "idx": "#", "label": "t1/t2", "start": "Start", "stop": "Stop",
            "n_frames": "Frames", "start_t": "Start (s)", "stop_t": "Stop (s)",
            "dur_s": "Dur (s)",
        }
        widths = {"idx": 40, "label": 55, "start": 80, "stop": 80, "n_frames": 80,
                  "start_t": 90, "stop_t": 90, "dur_s": 80}
        for c in cols:
            self.tree.heading(c, text=headings[c])
            self.tree.column(c, width=widths[c], anchor=tk.CENTER)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scroll.set)

        # Bottom action bar -------------------------------------------------
        action_frame = ttk.Frame(self.root, padding=6)
        action_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(action_frame, text="Delete Selected", command=self.delete_selected).pack(side=tk.LEFT)
        ttk.Button(action_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=6)
        ttk.Button(action_frame, text="Export CSV + Clips", command=self.export).pack(side=tk.RIGHT)

        # Status line -------------------------------------------------------
        self.status = ttk.Label(self.root, text="Ready.", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self.step(-1))
        self.root.bind("<Right>", lambda e: self.step(1))
        self.root.bind("<Shift-Left>", lambda e: self.step(-10))
        self.root.bind("<Shift-Right>", lambda e: self.step(10))
        self.root.bind("s", lambda e: self.set_start())
        self.root.bind("e", lambda e: self.set_stop())

    # -------------------------------------------------------------- video --
    def open_video_dialog(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v"),
                       ("All files", "*.*")],
        )
        if path:
            self.load_video(path)

    def load_video(self, path):
        if not os.path.isfile(path):
            messagebox.showerror("Error", f"File not found:\n{path}")
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open video:\n{path}")
            return

        # release previous
        if self.cap is not None:
            self.cap.release()

        self.cap = cap
        self.video_path = path
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0
        self.pending_start = None
        self.pairs.clear()
        self._refresh_tree()
        self._update_pending_label()

        self.slider.configure(to=max(self.total_frames - 1, 0))
        self.info_label.config(
            text=f"{os.path.basename(path)} | {self.frame_w}x{self.frame_h} "
                 f"| {self.fps:.2f} fps | {self.total_frames} frames"
        )
        self.show_frame(0)
        self.set_status(f"Loaded {os.path.basename(path)}")

    def _read_frame(self, idx):
        if self.cap is None:
            return None
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def show_frame(self, idx):
        if self.cap is None:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        frame = self._read_frame(idx)
        if frame is None:
            return
        self.current_frame_idx = idx

        # convert BGR -> RGB and scale for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = min(self.MAX_DISPLAY_W / w, self.MAX_DISPLAY_H / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        frame_resized = cv2.resize(frame_rgb, (disp_w, disp_h))

        img = Image.fromarray(frame_resized)
        self._photo = ImageTk.PhotoImage(image=img)

        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

        # overlay frame/time text
        t = idx / self.fps if self.fps else 0
        self.canvas.create_text(
            8, 8, anchor=tk.NW,
            text=f"Frame {idx}/{self.total_frames - 1}  |  {t:.3f}s",
            fill="yellow", font=("Helvetica", 12, "bold"),
        )

        # sync widgets without re-triggering the slider callback
        self.frame_var.set(idx)
        self.slider.set(idx)
        self.frame_entry.delete(0, tk.END)
        self.frame_entry.insert(0, str(idx))

    def step(self, delta):
        if self.cap is None:
            return
        self.show_frame(self.current_frame_idx + delta)

    def _on_slider(self, value):
        if self.cap is None:
            return
        idx = int(float(value))
        if idx != self.current_frame_idx:
            self.show_frame(idx)

    def _on_frame_entry(self, event=None):
        if self.cap is None:
            return
        try:
            idx = int(self.frame_entry.get())
        except ValueError:
            return
        self.show_frame(idx)

    # ----------------------------------------------------------- markers --
    def set_start(self):
        if self.cap is None:
            return
        self.pending_start = self.current_frame_idx
        self._update_pending_label()
        self.set_status(f"Start marked at frame {self.pending_start}")

    def set_stop(self):
        if self.cap is None:
            return
        if self.pending_start is None:
            messagebox.showwarning("No start", "Set a start frame first (press 's').")
            return
        start = self.pending_start
        stop = self.current_frame_idx
        if stop < start:
            start, stop = stop, start  # be forgiving about ordering

        label = self._ask_t1_t2()
        if label is None:
            # user cancelled — don't add the pair, keep the pending start
            self.set_status("Stop cancelled (no t1/t2 chosen). Pending start kept.")
            return

        self.pairs.append({"start": start, "stop": stop, "label": label})
        self.pending_start = None
        self._update_pending_label()
        self._refresh_tree()
        self.set_status(f"Added clip [{label}]: frames {start}–{stop}")

    def _ask_t1_t2(self):
        """Modal dialog asking whether this stop frame is t1 or t2.
        Returns 't1', 't2', or None if cancelled."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Label this time point")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        result = {"value": None}

        ttk.Label(dlg, text="Is this stop frame t1 or t2?",
                  padding=12).pack(side=tk.TOP)

        btn_frame = ttk.Frame(dlg, padding=(12, 0, 12, 12))
        btn_frame.pack(side=tk.TOP)

        def choose(val):
            result["value"] = val
            dlg.destroy()

        t1_btn = ttk.Button(btn_frame, text="t1", width=10,
                            command=lambda: choose("t1"))
        t1_btn.pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="t2", width=10,
                   command=lambda: choose("t2")).pack(side=tk.LEFT, padx=6)

        # keyboard shortcuts: press 1 / 2, or Esc to cancel
        dlg.bind("1", lambda e: choose("t1"))
        dlg.bind("2", lambda e: choose("t2"))
        dlg.bind("<Escape>", lambda e: dlg.destroy())

        # center over the main window
        dlg.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - dlg.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{max(x, 0)}+{max(y, 0)}")

        t1_btn.focus_set()
        self.root.wait_window(dlg)
        return result["value"]

    def _update_pending_label(self):
        txt = "--" if self.pending_start is None else str(self.pending_start)
        self.pending_label.config(text=f"Pending start: {txt}")

    # ------------------------------------------------------------- list ----
    def _refresh_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, p in enumerate(self.pairs):
            start, stop = p["start"], p["stop"]
            label = p.get("label", "")
            n = stop - start + 1
            st = start / self.fps if self.fps else 0
            et = stop / self.fps if self.fps else 0
            dur = (stop - start) / self.fps if self.fps else 0
            self.tree.insert(
                "", tk.END, iid=str(i),
                values=(i + 1, label, start, stop, n,
                        f"{st:.3f}", f"{et:.3f}", f"{dur:.3f}"),
            )

    def delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        idxs = sorted((int(s) for s in sel), reverse=True)
        for i in idxs:
            del self.pairs[i]
        self._refresh_tree()
        self.set_status(f"Deleted {len(idxs)} clip(s)")

    def clear_all(self):
        if not self.pairs:
            return
        if messagebox.askyesno("Clear all", "Remove all annotated pairs?"):
            self.pairs.clear()
            self._refresh_tree()
            self.set_status("Cleared all pairs")

    # ----------------------------------------------------------- export ----
    def export(self):
        if self.cap is None or self.video_path is None:
            messagebox.showwarning("No video", "Load a video first.")
            return
        if not self.pairs:
            messagebox.showwarning("No pairs", "Annotate at least one clip first.")
            return

        out_dir = filedialog.askdirectory(title="Choose output folder")
        if not out_dir:
            return

        base = os.path.splitext(os.path.basename(self.video_path))[0]
        csv_path = os.path.join(out_dir, f"{base}_clips.csv")

        # Write CSV first
        try:
            self._write_csv(csv_path, base)
        except Exception as exc:
            messagebox.showerror("CSV error", str(exc))
            return

        # Extract clips in a background thread so the GUI stays responsive
        self.set_status("Exporting clips... (this may take a while)")
        threading.Thread(
            target=self._extract_clips_thread,
            args=(out_dir, base, csv_path),
            daemon=True,
        ).start()

    # Mice in the control group; everything else is treated as SNr-DTA.
    CTRL_IDS = {"SC01", "SC07", "SC08", "SC13", "SC14", "SC33", "SC34", "SC35", "SC36"}

    def _parse_metadata(self, base):
        """Derive ID, Day, and Group from the source video base filename.
        ID    = first 4 characters.
        Day   = 4th character counting from the end, as an int (blank if not a digit).
        Group = 'Ctrl' if ID in CTRL_IDS else 'SNr-DTA'.
        """
        vid_id = base[:4]

        day = ""
        if len(base) >= 5:
            ch = base[-5]
            if ch.isdigit():
                day = int(ch)
            elif base[-11:].isdigit():
                day = base[-11:]


        group = "Ctrl" if vid_id in self.CTRL_IDS else "SNr-DTA"
        return vid_id, day, group

    def _write_csv(self, csv_path, base):
        vid_id, day, group = self._parse_metadata(base)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "clip_index", "ID", "Day", "Group", "time_point",
                "clip_filename", "source_video",
                "start_frame", "stop_frame", "n_frames",
                "start_time_s", "stop_time_s", "duration_s", "fps",
            ])
            for i, p in enumerate(self.pairs):
                start, stop = p["start"], p["stop"]
                label = p.get("label", "")
                n = stop - start + 1
                st = start / self.fps if self.fps else 0
                et = stop / self.fps if self.fps else 0
                dur = (stop - start) / self.fps if self.fps else 0
                clip_name = f"{base}_clip{i + 1:03d}_{label}_f{start}-{stop}.mp4"
                writer.writerow([
                    i + 1, vid_id, day, group, label,
                    clip_name, os.path.basename(self.video_path),
                    start, stop, n,
                    f"{st:.4f}", f"{et:.4f}", f"{dur:.4f}", f"{self.fps:.4f}",
                ])

    def _extract_clips_thread(self, out_dir, base, csv_path):
        # Use a dedicated capture object for thread safety
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        total = len(self.pairs)

        try:
            for i, p in enumerate(self.pairs):
                start, stop = p["start"], p["stop"]
                label = p.get("label", "")
                clip_name = f"{base}_clip{i + 1:03d}_{label}_f{start}-{stop}.mp4"
                clip_path = os.path.join(out_dir, clip_name)

                writer = cv2.VideoWriter(
                    clip_path, fourcc, self.fps, (self.frame_w, self.frame_h)
                )
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for fidx in range(start, stop + 1):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    writer.write(frame)
                writer.release()

                self._set_status_threadsafe(
                    f"Exported clip {i + 1}/{total}: {clip_name}"
                )
        finally:
            cap.release()

        self._set_status_threadsafe(
            f"Done. CSV + {total} clip(s) saved to {out_dir}"
        )
        self.root.after(0, lambda: messagebox.showinfo(
            "Export complete",
            f"Saved CSV:\n{csv_path}\n\nand {total} clip(s) to:\n{out_dir}",
        ))

    # ----------------------------------------------------------- helpers ---
    def set_status(self, text):
        self.status.config(text=text)

    def _set_status_threadsafe(self, text):
        self.root.after(0, lambda: self.set_status(text))

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    app = VideoClipAnnotator(root, initial_video=initial)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
