# Y-Maze Analyzer

A Python toolkit for annotating, labeling, and statistically analyzing reward-based Y-maze behavioral videos in mice.

**Experiment summary:** Mice (SNr-DTA lesion vs. control) are food-restricted to 85% body weight and trained over 5 days to choose the baited arm of a Y-maze. Each session is recorded, clipped into individual trials (t1 = approach, t2 = consumption), and analyzed for group differences in decision latency and accuracy.

**Dependencies:** `opencv-python`, `pillow`, `pandas`, `numpy`, `scipy`, `statsmodels`, `ffmpeg` (for `video_concat_gui.py`)

```bash
pip install opencv-python pillow pandas numpy scipy statsmodels
```

---

## Pipeline Overview

```
video_concat_gui.py          (optional — merge multi-part recordings)
        │
        ▼
step1_video_clip_annotator.py    annotate t1/t2 clips from video
        │
        ├── step1a_relabel.py    (optional — fix mouse IDs / group labels)
        │
        ├── step1b_scan.py       (optional — batch scan all CSVs for mislabeled frames)
        │
        └── step1b_fix_clips.py  (optional — GUI to fix individual clips)
        │
        ▼
step2_generate_log_raw.py        concatenate all *_clips.csv into one log
        │
        ▼
step3_label_success_or_failure.py    manually label each trial as hit/miss
        │
        ▼
step4_statistics.py              compute per-day duration statistics
        │
        ▼
step5_check_significance.py      LMM + Mann-Whitney group comparison
```

> **Folder convention:** every step reads and writes in the **same folder** as its input files.

---

## Step-by-Step

### `video_concat_gui.py` *(optional)*

| | |
|---|---|
| **Input** | Multiple raw video files (`.mp4`, etc.) — selected via GUI |
| **Output** | One merged `.mp4` — saved to user-chosen path |
| **Requires** | `ffmpeg` on system PATH |

---

### Step 1 — `step1_video_clip_annotator.py`

Scrub through the source video, mark start/stop frames for each t1 and t2 clip, and export.

| | |
|---|---|
| **Input** | A single video file — loaded via GUI |
| **Output** | `<videoname>_clips.csv` — one row per clip (start/stop frames, timestamps, mouse ID, day, label) |
| **Output** | Extracted `.mp4` clip files for each annotated segment |
| **Where** | Same folder as the source video |

**Controls:** `s` = set start, `e` = set stop, `←/→` = step 1 frame, `Shift+←/→` = step 10 frames

---

### Step 1a — `step1a_relabel.py` *(optional)*

Fix mislabeled mouse IDs or group assignments across all CSVs in a folder.

| | |
|---|---|
| **Input** | All `*_clips.csv`, `ymaze_time_log_raw.csv`, `ymaze_time_log_labeled.csv` in a folder |
| **Output** | Same files with corrected IDs/groups (timestamped backups saved first) |

---

### Step 1b — `step1b_scan.py` *(optional — run first)*

Batch scan **all** `*_clips.csv` files in a folder at once to find suspicious clips.

**Flags:** a T2 clip whose `stop_frame` is within ±100 frames of the next clip's `start_frame` (the specific mislabeling pattern where the t2 end was accidentally set at the next t1's boundary).

```bash
python step1b_scan.py /path/to/clips/folder
python step1b_scan.py /path/to/clips/folder --threshold 50   # stricter
```

| | |
|---|---|
| **Input** | All `*_clips.csv` in a folder |
| **Output** | Console report grouped by file + `suspicious_clips_report.csv` |

---

### Step 1b — `step1b_fix_clips.py` *(optional — fix after scanning)*

GUI tool to fix individual clips without re-annotating the whole video.

| | |
|---|---|
| **Input** | A single `*_clips.csv` + optionally the source video |
| **Output** | Updated `*_clips.csv` with corrected frames/labels; re-extracted clip `.mp4` for changed rows |
| **Flags** | Suspicious rows highlighted in red (same T2-stop heuristic as the scan) |

**Workflow:** Open CSV → suspicious rows appear in red → click a row → video jumps to that region → scrub to correct frame → "Set Stop = current frame" → "Fix Selected"

---

### Step 2 — `step2_generate_log_raw.py`

| | |
|---|---|
| **Input** | All `*_clips.csv` files in a folder |
| **Output** | `ymaze_time_log_without_hitmisslabel.csv` — concatenated log with `Day` column |
| **Output** | `ymaze_time_stats_alltrials.csv` — per-day t1/t2 duration statistics |

---

### Step 3 — `step3_label_success_or_failure.py`

GUI popup per (mouse, day) session — check boxes to mark each trial as correct or not.

| | |
|---|---|
| **Input** | `ymaze_time_log_without_hitmisslabel.csv` |
| **Output** | `ymaze_time_log_labeled.csv` — same data + `correct` column (1 = hit, 0 = miss) |

---

### Step 4 — `step4_statistics.py`

| | |
|---|---|
| **Input** | All `*_clips.csv` + `ymaze_time_log_labeled.csv` |
| **Output** | `ymaze_time_stats.csv` — all trials |
| **Output** | `ymaze_time_stats_hits.csv` — correct trials only |
| **Output** | `ymaze_time_stats_misses.csv` — incorrect trials only |

---

### Step 5 — `step5_check_significance.py`

Compares SNr-DTA vs Control group duration for T1, T2, and T1+T2 combined using Linear Mixed Models and per-day Mann-Whitney U tests.

```bash
python step5_check_significance.py /path/to/folder
python step5_check_significance.py /path/to/folder --subset hits
python step5_check_significance.py /path/to/folder --subset misses
```

| | |
|---|---|
| **Input** | `ymaze_time_log_raw.csv` or `ymaze_time_log_labeled.csv` |
| **Output** | `ymaze_stats_results.csv` — per-day Mann-Whitney U results |
| **Output** | `ymaze_stats_lmm.txt` — LMM summary tables |
| **Output** | Console printout of all key results |

---

## CSV File Lineage

```
*_clips.csv  (one per recording session, from Step 1)
    │
    ├── step1b_fix_clips.py (optional) ──► corrected *_clips.csv
    │
    ├── step2 ──► ymaze_time_log_without_hitmisslabel.csv
    │                   │
    │                   └── step3 ──► ymaze_time_log_labeled.csv
    │                                       │
    └── step4 (uses both) ─────────────────►│
              │                             │
              ▼                             ▼
    ymaze_time_stats.csv          ymaze_stats_results.csv  (step5)
    ymaze_time_stats_hits.csv     ymaze_stats_lmm.txt      (step5)
    ymaze_time_stats_misses.csv
```
