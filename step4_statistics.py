#!/usr/bin/env python3
"""
concat_clips_and_stats.py
=========================

Looks for files matching ``*_clips.csv`` in a folder, concatenates them,
fills the ``Day`` column, and computes per-day t1/t2 duration statistics.

If ``ymaze_time_log_labeled.csv`` exists in the folder (produced by
``label_trial_hits.py``), the ``correct`` column from it is merged in and
two additional stats files are produced:

    ymaze_time_stats_hits.csv   — only correct == 1 trials
    ymaze_time_stats_misses.csv — only correct == 0 trials

If the labeled file is absent, only the original combined stats are saved.

Dependencies:
    pip install pandas

Usage:
    python concat_clips_and_stats.py                # uses current directory
    python concat_clips_and_stats.py /path/to/dir   # explicit folder
"""

import os
import re
import sys
import glob

import pandas as pd


# ── file names ─────────────────────────────────────────────────────────────────
INPUT_GLOB        = "*_clips.csv"
LABELED_FILE      = "ymaze_time_log_labeled.csv"   # output of label_trial_hits.py
RAW_OUT_NAME      = "ymaze_time_log_raw_JCincluded.csv"
STATS_OUT_NAME    = "ymaze_time_stats.csv"           # all trials (unchanged)
STATS_HITS_NAME   = "ymaze_time_stats_hits.csv"      # correct == 1
STATS_MISSES_NAME = "ymaze_time_stats_misses.csv"    # correct == 0

EXPECTED_TRIALS_PER_DAY = 5

DAY_PATTERN = re.compile(r"day(\d+)(?=\D|$)", re.IGNORECASE)


# ── helpers ────────────────────────────────────────────────────────────────────

def extract_day(clip_filename):
    if not isinstance(clip_filename, str):
        return pd.NA
    m = DAY_PATTERN.search(clip_filename)
    return int(m.group(1)) - 4 if m else pd.NA


def find_input_files(in_dir):
    files = sorted(glob.glob(os.path.join(in_dir, INPUT_GLOB)))
    skip_prefixes = (
        RAW_OUT_NAME.rsplit(".", 1)[0],
        STATS_OUT_NAME.rsplit(".", 1)[0],
        STATS_HITS_NAME.rsplit(".", 1)[0],
        STATS_MISSES_NAME.rsplit(".", 1)[0],
        LABELED_FILE.rsplit(".", 1)[0],
    )
    return [f for f in files if not os.path.basename(f).startswith(skip_prefixes)]


def fix_sc01_to_sc10(filepath):
    """Fix labelling bug: SC01 rows actually belong to SC10."""
    df = pd.read_csv(filepath)
    if "ID" not in df.columns:
        return 0
    mask = df["ID"].astype(str) == "SC01"
    n = int(mask.sum())
    if n == 0:
        return 0
    df.loc[mask, "ID"] = "SC10"
    if "Group" in df.columns:
        df.loc[mask, "Group"] = "SNr-DTA"
    df.to_csv(filepath, index=False)
    return n


def load_and_concat(files):
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            print(f"  ! Skipping {os.path.basename(f)}: {exc}")
            continue
        df["source_csv"] = os.path.basename(f)
        frames.append(df)
    if not frames:
        sys.exit("No readable CSVs.")
    return pd.concat(frames, ignore_index=True)


def compute_stats(raw_df):
    """
    Aggregate mean/std/n duration per (ID, Day, Group, time_point), wide format.
    Also adds performance columns per mouse-day:
        n_hits          — number of correct==1 trials  (counted via t1 rows only
                          since both t1 and t2 of a trial share the same label)
        n_trials_total  — total number of trials (t1 row count)
        pct_correct     — n_hits / n_trials_total * 100
    If the 'correct' column is absent, these columns are filled with NaN.
    """
    needed = {"ID", "Day", "Group", "time_point", "duration_s", "x_split"}
    missing = needed - set(raw_df.columns)
    if missing:
        sys.exit(f"Missing columns for stats: {sorted(missing)}")

    df = raw_df.copy()
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    df = df.dropna(subset=["Day", "time_point", "duration_s"])

    long_stats = (
        df.groupby(["ID", "Day", "Group", "time_point"], dropna=False)
        .agg(
            mean_duration_s=("duration_s", "mean"),
            std_duration_s =("duration_s", "std"),
            n_trials       =("duration_s", "count"),
        )
        .reset_index()
    )

    wide = long_stats.pivot_table(
        index=["ID", "Day", "Group"],
        columns="time_point",
        values=["mean_duration_s", "std_duration_s", "n_trials"],
    )
    wide.columns = [f"{metric}_{tp}" for metric, tp in wide.columns]

    ordered = []
    for metric in ("mean_duration_s", "std_duration_s", "n_trials"):
        for tp in ("t1", "t2"):
            col = f"{metric}_{tp}"
            if col in wide.columns:
                ordered.append(col)

    wide = wide[ordered].reset_index().sort_values(["ID", "Day"])
    wide["Day"] = pd.to_numeric(wide["Day"], errors="coerce").astype("Int64")

    # ── Performance score ──────────────────────────────────────────────────────
    # Use t1 rows only — each trial has exactly one t1, so this counts unique
    # trials without double-counting the paired t2.
    if "correct" in raw_df.columns:
        t1_rows = raw_df[raw_df["time_point"] == "t1"].copy()
        t1_rows["correct"] = pd.to_numeric(t1_rows["correct"], errors="coerce")
        perf = (
            t1_rows.groupby(["ID", "Day"])
            .agg(
                n_hits         =("correct", lambda x: (x == 1).sum()),
                n_trials_total =("correct", "count"),
            )
            .reset_index()
        )
        perf["pct_correct"] = (perf["n_hits"] / perf["n_trials_total"] * 100).round(1)
        perf["Day"] = pd.to_numeric(perf["Day"], errors="coerce").astype("Int64")
        wide = wide.merge(perf, on=["ID", "Day"], how="left")
    else:
        wide["n_hits"]          = pd.NA
        wide["n_trials_total"]  = pd.NA
        wide["pct_correct"]     = pd.NA

    return wide


def save_stats(df, path, label):
    """Compute stats on df and save; print summary."""
    if df.empty:
        print(f"  ! Skipping {label} — no rows.")
        return
    stats = compute_stats(df)
    stats["x_split"] = stats["Day"]
    stats.to_csv(path, index=False)
    print(f"Saved {label:30s}: {path}  ({len(stats)} mouse-day rows, "
          f"{int(df['duration_s'].notna().sum())} clips)")

    # sanity check trial counts
    for col in ("n_trials_t1", "n_trials_t2"):
        if col in stats.columns:
            off = stats[stats[col].fillna(0) != EXPECTED_TRIALS_PER_DAY]
            if len(off):
                print(f"    ! {len(off)} mouse-day row(s) have {col} != "
                      f"{EXPECTED_TRIALS_PER_DAY}:")
                print("    " + off[["ID", "Day", "Group", col]]
                      .to_string(index=False).replace("\n", "\n    "))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Extreme SSD/ymaze/t1t2_and_clips"
    if not os.path.isdir(in_dir):
        sys.exit(f"Not a directory: {in_dir}")

    # ── 1. Find and load per-session clip CSVs ─────────────────────────────────
    files = find_input_files(in_dir)
    if not files:
        sys.exit(f"No files matching '{INPUT_GLOB}' found in {in_dir}")

    print(f"Found {len(files)} clip CSV file(s):")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # ── 2. SC01 → SC10 correction ──────────────────────────────────────────────
    print("\nChecking for SC01 -> SC10 corrections...")
    total_fixed = 0
    for f in files:
        n_fixed = fix_sc01_to_sc10(f)
        if n_fixed:
            total_fixed += n_fixed
            print(f"  ✎ {os.path.basename(f)}: rewrote {n_fixed} row(s) "
                  f"SC01 -> SC10 (Group -> SNr-DTA)")
    if total_fixed == 0:
        print("  (none found)")

    # ── 3. Concatenate and fill Day ────────────────────────────────────────────
    combined = load_and_concat(files)

    if "clip_filename" not in combined.columns:
        sys.exit("Inputs are missing a 'clip_filename' column.")

    combined["Day"]     = combined["clip_filename"].apply(extract_day)
    combined["Day"]     = pd.to_numeric(combined["Day"], errors="coerce").astype("Int64")
    combined["x_split"] = combined["Day"]

    n_missing = combined["Day"].isna().sum()
    if n_missing:
        print(f"  ! {n_missing} row(s) had no 'dayN_' match in clip_filename")

    # ── 4. Save raw combined ───────────────────────────────────────────────────
    raw_out = os.path.join(in_dir, RAW_OUT_NAME)
    combined.to_csv(raw_out, index=False)
    print(f"\nSaved raw combined CSV  : {raw_out}  ({len(combined)} rows)")

    # ── 5. Merge correct labels if labeled file exists ─────────────────────────
    labeled_path = os.path.join(in_dir, LABELED_FILE)
    has_labels   = os.path.exists(labeled_path)

    if has_labels:
        print(f"\nFound labeled file: {LABELED_FILE}")
        labeled_df = pd.read_csv(labeled_path)

        if "correct" not in labeled_df.columns:
            print("  ! 'correct' column missing from labeled file — skipping hit/miss split.")
            has_labels = False
        else:
            # merge on the minimal key set that uniquely identifies each row
            merge_keys = ["ID", "Day", "clip_filename", "time_point", "start_frame"]
            merge_keys = [k for k in merge_keys if k in labeled_df.columns
                          and k in combined.columns]

            combined = combined.merge(
                labeled_df[merge_keys + ["correct"]],
                on=merge_keys,
                how="left",
            )
            n_labeled = combined["correct"].notna().sum()
            print(f"  Merged 'correct' labels: {n_labeled} / {len(combined)} rows have a label.")
    else:
        print(f"\nNo labeled file found ({LABELED_FILE}) — only combined stats will be saved.")
        print("  Run label_trial_hits.py first to enable hit/miss split.")

    # ── 6. Save stats ──────────────────────────────────────────────────────────
    print()

    # 6a. All trials (original behaviour, always produced)
    save_stats(
        combined,
        os.path.join(in_dir, STATS_OUT_NAME),
        "all trials",
    )

    if has_labels:
        # 6b. Hits only (correct == 1)
        hits_df = combined[combined["correct"] == 1].copy()
        save_stats(
            hits_df,
            os.path.join(in_dir, STATS_HITS_NAME),
            "correct hits  (correct=1)",
        )

        # 6c. Misses only (correct == 0)
        misses_df = combined[combined["correct"] == 0].copy()
        save_stats(
            misses_df,
            os.path.join(in_dir, STATS_MISSES_NAME),
            "misses        (correct=0)",
        )

        unlabeled = combined["correct"].isna().sum()
        if unlabeled:
            print(f"\n  Note: {unlabeled} row(s) have no label (NaN) and are "
                  "excluded from both hits and misses files.")


if __name__ == "__main__":
    main()