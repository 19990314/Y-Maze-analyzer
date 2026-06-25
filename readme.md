# Y-Maze Analyzer — File Workflow

---

## Overview

```
video_concat_gui.py  (optional)
        │
        ▼
step1_video_clip_annotator.py
        │
        ▼
step1a_relabel.py  (optional)
        │
        ▼
step2_generate_log_raw.py
        │
        ▼
step3_label_success_or_failure.py
        │
        ▼
step4_statistics.py
        │
        ▼
step5_check_significance.py
```

> **Folder convention:** every step reads and writes in the **same folder** as its input files.

---

## Step-by-Step

### `video_concat_gui.py` *(optional)*

| | |
|---|---|
| **Input** | Multiple raw video files (`.mp4`, etc.) — selected via GUI |
| **Output** | One merged `.mp4` file — saved to user-chosen path |
| **Requires** | `ffmpeg` installed on system PATH |

---

### Step 1 — `step1_video_clip_annotator.py`

| | |
|---|---|
| **Input** | A single video file — loaded via GUI |
| **Output** | `<videoname>_clips.csv` — one row per annotated clip (start/stop frames, timestamps, mouse ID, day) |
| **Output** | Extracted video clip files (`.mp4`) for each segment |
| **Where** | Same folder as the source video |

---

### Step 1a — `step1a_relabel.py` *(optional)*

| | |
|---|---|
| **Input** | `ymaze_time_log_raw.csv`, `ymaze_time_log_labeled.csv`, `*_clips.csv` — all in a user-selected folder |
| **Output** | Same files with corrected mouse IDs and/or group labels |
| **Note** | Timestamped backups are saved before any changes are written |
| **Where** | Same folder as input CSVs |

---

### Step 2 — `step2_generate_log_raw.py`

| | |
|---|---|
| **Input** | All `*_clips.csv` files in a folder |
| **Output** | `ymaze_time_log_without_hitmisslabel.csv` — concatenated clips with `Day` column filled in |
| **Output** | `ymaze_time_stats_alltrials.csv` — per-day t1/t2 duration statistics |
| **Where** | Same folder as the `*_clips.csv` files |

---

### Step 3 — `step3_label_success_or_failure.py`

| | |
|---|---|
| **Input** | `ymaze_time_log_without_hitmisslabel.csv` |
| **Output** | `ymaze_time_log_labeled.csv` — same data with a new `correct` column (1 = hit, 0 = miss) |
| **How** | GUI popup appears for each (mouse, day) session — user checks trial-by-trial boxes |
| **Where** | Same folder as input |

---

### Step 4 — `step4_statistics.py`

| | |
|---|---|
| **Input** | All `*_clips.csv` files + `ymaze_time_log_labeled.csv` |
| **Output** | `ymaze_time_stats.csv` — statistics for all trials |
| **Output** | `ymaze_time_stats_hits.csv` — statistics for correct trials only |
| **Output** | `ymaze_time_stats_misses.csv` — statistics for incorrect trials only |
| **Where** | Same folder as input |

---

### Step 5 — `step5_check_significance.py`

| | |
|---|---|
| **Input** | `ymaze_time_log_raw.csv` or `ymaze_time_log_labeled.csv` |
| **Input** | Optional flag: `--subset hits` / `misses` / `all` (default: `all`) |
| **Output** | `ymaze_stats_results.csv` — per-day Mann-Whitney U results |
| **Output** | `ymaze_stats_lmm.txt` — Linear Mixed Model summary tables |
| **Output** | Console printout of all key statistical results |
| **Where** | Same folder as input |

---

## CSV File Lineage

```
*_clips.csv  (one per recording session, from Step 1)
    │
    ├──► step2 ──► ymaze_time_log_without_hitmisslabel.csv
    │                   │
    │                   └──► step3 ──► ymaze_time_log_labeled.csv
    │                                       │
    └──► step4 (uses both) ─────────────────┤
              │                             │
              ▼                             ▼
    ymaze_time_stats.csv          ymaze_stats_results.csv  (step5)
    ymaze_time_stats_hits.csv     ymaze_stats_lmm.txt      (step5)
    ymaze_time_stats_misses.csv
```
