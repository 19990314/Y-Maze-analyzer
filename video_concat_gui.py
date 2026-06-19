"""
video_concat_gui.py
-------------------
A Tkinter GUI to select multiple video files and concatenate them
into a single output file using ffmpeg.

Requirements:
    pip install tkinter  (usually built-in)
    ffmpeg must be installed and on PATH:
        macOS:   brew install ffmpeg
        Ubuntu:  sudo apt install ffmpeg
        Windows: https://ffmpeg.org/download.html
"""

import os
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading


# ── helpers ────────────────────────────────────────────────────────────────────

def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def move_item(listbox: tk.Listbox, direction: int) -> None:
    """Move selected listbox item up (-1) or down (+1)."""
    sel = listbox.curselection()
    if not sel:
        return
    idx = sel[0]
    new_idx = idx + direction
    if new_idx < 0 or new_idx >= listbox.size():
        return
    text = listbox.get(idx)
    listbox.delete(idx)
    listbox.insert(new_idx, text)
    listbox.select_set(new_idx)
    listbox.see(new_idx)


# ── main app ───────────────────────────────────────────────────────────────────

class VideoConcatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Concatenator")
        self.resizable(True, True)
        self.minsize(620, 480)
        self.configure(bg="#1e1e2e")

        # colour palette
        self.BG       = "#1e1e2e"
        self.PANEL    = "#2a2a3e"
        self.ACCENT   = "#7c6af7"
        self.ACCENT2  = "#5a4fcf"
        self.FG       = "#e0e0f0"
        self.FG_DIM   = "#888899"
        self.RED      = "#f07070"
        self.GREEN    = "#70c070"

        self._build_ui()

        if not check_ffmpeg():
            messagebox.showwarning(
                "ffmpeg not found",
                "ffmpeg was not found on your PATH.\n\n"
                "macOS:   brew install ffmpeg\n"
                "Ubuntu:  sudo apt install ffmpeg\n"
                "Windows: https://ffmpeg.org/download.html"
            )

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        # ── title ──
        tk.Label(
            self, text="🎬  Video Concatenator",
            font=("Helvetica", 16, "bold"),
            bg=self.BG, fg=self.ACCENT
        ).pack(pady=(18, 4))

        tk.Label(
            self, text="Select videos, reorder them, then export.",
            font=("Helvetica", 10), bg=self.BG, fg=self.FG_DIM
        ).pack(pady=(0, 10))

        # ── file list panel ──
        list_frame = tk.Frame(self, bg=self.PANEL, bd=0, relief="flat")
        list_frame.pack(fill="both", expand=True, **pad)

        tk.Label(
            list_frame, text="Video files (drag to reorder with buttons →)",
            font=("Helvetica", 9), bg=self.PANEL, fg=self.FG_DIM
        ).pack(anchor="w", padx=8, pady=(6, 2))

        inner = tk.Frame(list_frame, bg=self.PANEL)
        inner.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        scrollbar = tk.Scrollbar(inner, bg=self.PANEL)
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            inner,
            yscrollcommand=scrollbar.set,
            selectmode="single",
            bg="#13131f", fg=self.FG,
            selectbackground=self.ACCENT, selectforeground="white",
            font=("Courier", 10),
            relief="flat", bd=0,
            highlightthickness=0,
            activestyle="none",
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        # ── file buttons ──
        btn_row = tk.Frame(self, bg=self.BG)
        btn_row.pack(fill="x", padx=12, pady=(0, 4))

        self._btn(btn_row, "＋  Add videos",  self.add_videos).pack(side="left", padx=(0, 6))
        self._btn(btn_row, "✕  Remove",       self.remove_selected, danger=True).pack(side="left", padx=(0, 6))
        self._btn(btn_row, "▲  Up",           lambda: move_item(self.listbox, -1)).pack(side="left", padx=(0, 6))
        self._btn(btn_row, "▼  Down",         lambda: move_item(self.listbox, +1)).pack(side="left")
        self._btn(btn_row, "🗑  Clear all",    self.clear_all, danger=True).pack(side="right")

        # ── output filename ──
        out_frame = tk.Frame(self, bg=self.PANEL, bd=0)
        out_frame.pack(fill="x", padx=12, pady=(4, 6))

        tk.Label(
            out_frame, text="Output filename:",
            font=("Helvetica", 10, "bold"),
            bg=self.PANEL, fg=self.FG
        ).pack(side="left", padx=(10, 8), pady=10)

        self.output_var = tk.StringVar(value="output.mp4")
        self.output_entry = tk.Entry(
            out_frame, textvariable=self.output_var,
            font=("Courier", 10),
            bg="#13131f", fg=self.FG,
            insertbackground=self.FG,
            relief="flat", bd=4,
            width=30,
        )
        self.output_entry.pack(side="left", pady=10)

        self._btn(out_frame, "Browse…", self.browse_output).pack(side="left", padx=8, pady=10)

        # ── progress bar ──
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=300)
        self.progress.pack(pady=(2, 0))

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "TProgressbar",
            troughcolor=self.PANEL,
            background=self.ACCENT,
            thickness=6,
        )

        # ── status label ──
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(
            self, textvariable=self.status_var,
            font=("Helvetica", 9), bg=self.BG, fg=self.FG_DIM
        ).pack(pady=(4, 2))

        # ── concat button ──
        self.concat_btn = self._btn(
            self, "▶  Concatenate videos",
            self.start_concat,
            big=True
        )
        self.concat_btn.pack(pady=(4, 16), ipadx=12, ipady=4)

    def _btn(self, parent, text, cmd, danger=False, big=False):
        color  = self.RED    if danger else (self.ACCENT if not big else self.ACCENT)
        hover  = "#c05050"   if danger else self.ACCENT2
        fg     = "white"
        font   = ("Helvetica", 11, "bold") if big else ("Helvetica", 9)

        b = tk.Button(
            parent, text=text, command=cmd,
            bg=color, fg=fg,
            activebackground=hover, activeforeground=fg,
            font=font, relief="flat", bd=0, cursor="hand2",
            padx=10, pady=4,
        )
        b.bind("<Enter>", lambda e: b.config(bg=hover))
        b.bind("<Leave>", lambda e: b.config(bg=color))
        return b

    # ── actions ───────────────────────────────────────────────────────────────

    def add_videos(self):
        files = filedialog.askopenfilenames(
            title="Select video files",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v"),
                ("All files", "*.*"),
            ]
        )
        for f in files:
            self.listbox.insert("end", f)

    def remove_selected(self):
        sel = self.listbox.curselection()
        if sel:
            self.listbox.delete(sel[0])

    def clear_all(self):
        if self.listbox.size() and messagebox.askyesno("Clear all", "Remove all videos from the list?"):
            self.listbox.delete(0, "end")

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save output video as…",
            defaultextension=".mp4",
            filetypes=[
                ("MP4", "*.mp4"),
                ("AVI", "*.avi"),
                ("MOV", "*.mov"),
                ("MKV", "*.mkv"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self.output_var.set(path)

    # ── concat logic ──────────────────────────────────────────────────────────

    def start_concat(self):
        files = list(self.listbox.get(0, "end"))
        if len(files) < 2:
            messagebox.showwarning("Not enough files", "Please add at least 2 video files.")
            return

        output = self.output_var.get().strip()
        if not output:
            messagebox.showwarning("No output name", "Please enter an output filename.")
            return

        # ask for output name via dialog if entry is still the default / empty
        if output == "output.mp4":
            ans = messagebox.askyesno(
                "Output filename",
                f'Use "{output}" as the output filename?\n\n'
                "Click No to choose a different name."
            )
            if not ans:
                self.browse_output()
                output = self.output_var.get().strip()
                if not output:
                    return

        self.concat_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Concatenating… please wait.")

        thread = threading.Thread(target=self._run_ffmpeg, args=(files, output), daemon=True)
        thread.start()

    def _run_ffmpeg(self, files: list[str], output: str):
        try:
            # write a temp concat list file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                for f in files:
                    # ffmpeg concat demuxer needs escaped paths
                    escaped = f.replace("'", "'\\''")
                    tmp.write(f"file '{escaped}'\n")
                tmp_path = tmp.name

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", tmp_path,
                "-c", "copy",
                output
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            os.unlink(tmp_path)

            if result.returncode == 0:
                self.after(0, self._on_success, output)
            else:
                self.after(0, self._on_error, result.stderr)

        except Exception as exc:
            self.after(0, self._on_error, str(exc))

    def _on_success(self, output: str):
        self.progress.stop()
        self.concat_btn.config(state="normal")
        self.status_var.set(f"✓  Saved to: {output}")
        messagebox.showinfo(
            "Done!",
            f"Videos concatenated successfully!\n\nSaved to:\n{output}"
        )

    def _on_error(self, msg: str):
        self.progress.stop()
        self.concat_btn.config(state="normal")
        self.status_var.set("✗  Error occurred — see details.")
        messagebox.showerror(
            "ffmpeg error",
            f"Concatenation failed:\n\n{msg[:800]}"
        )


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = VideoConcatApp()
    app.mainloop()
