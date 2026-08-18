#!/usr/bin/env python3
"""
step1b_scan.py
==============
Scan all *_clips.csv files in a folder and print a report of every
suspicious row — where a clip's stop_frame is within THRESHOLD frames
of any adjacent clip's start or stop frame.

Usage:
    python step1b_scan.py                  # scans current directory
    python step1b_scan.py /path/to/folder
    python step1b_scan.py --threshold 50   # change frame tolerance (default 100)

Output:
    Console report grouped by file.
    Also saves  suspicious_clips_report.csv  in the scanned folder.
"""

import os
import sys
import glob
import argparse

import pandas as pd

THRESHOLD = 100


def flag_suspicious(df, threshold):
    """Flag rows where a T2 clip's stop_frame is within threshold frames of
    the next clip's start_frame.  T1→T2 adjacency is intentionally excluded
    because T2 always starts right after T1 ends (natural temporal sequence)."""
    rows = []
    labels = df["time_point"].astype(str).str.strip().str.lower().tolist()
    stops  = pd.to_numeric(df["stop_frame"],  errors="coerce").tolist()
    starts = pd.to_numeric(df["start_frame"], errors="coerce").tolist()

    for i in range(len(df) - 1):
        if labels[i] != "t2":
            continue
        t2_stop      = stops[i]
        next_start   = starts[i + 1]
        if pd.isna(t2_stop) or pd.isna(next_start):
            continue
        dist = abs(t2_stop - next_start)
        if 0 < dist <= threshold:
            row = df.iloc[i].to_dict()
            row["_row_in_file"] = i + 2
            row["_close_to"]    = f"next row {i + 3} start ({next_start:.0f})"
            row["_distance"]    = int(dist)
            rows.append(row)
    return rows


def scan_folder(folder, threshold):
    csvs = sorted(glob.glob(os.path.join(folder, "*_clips.csv")))
    if not csvs:
        print(f"No *_clips.csv files found in: {folder}")
        return

    all_suspects = []
    total_rows = 0

    for path in csvs:
        fname = os.path.basename(path)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")
            continue

        total_rows += len(df)
        suspects = flag_suspicious(df, threshold)

        if suspects:
            print(f"\n{'─'*60}")
            print(f"  {fname}  ({len(suspects)} suspicious)")
            print(f"{'─'*60}")
            for s in suspects:
                clip  = s.get("clip_filename", s.get("clip_index", "?"))
                start = s.get("start_frame", "?")
                stop  = s.get("stop_frame",  "?")
                label = s.get("time_point",  "?")
                close = s["_close_to"]
                dist  = s["_distance"]
                print(f"  row {s['_row_in_file']:>3}  [{label}]  "
                      f"start={start}  stop={stop}  "
                      f"→ within {dist} frames of {close}")
                print(f"         {clip}")
            for s in suspects:
                s["_source_file"] = fname
            all_suspects.extend(suspects)
        else:
            print(f"  OK  {fname}")

    print(f"\n{'='*60}")
    print(f"Scanned {len(csvs)} files, {total_rows} total clips.")
    print(f"Suspicious rows: {len(all_suspects)}")

    if all_suspects:
        report_path = os.path.join(folder, "suspicious_clips_report.csv")
        report_df = pd.DataFrame(all_suspects)
        # put the meta columns first
        meta = ["_source_file", "_row_in_file", "_close_to", "_distance"]
        other = [c for c in report_df.columns if c not in meta]
        report_df = report_df[meta + other]
        report_df.to_csv(report_path, index=False)
        print(f"Report saved → {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Scan *_clips.csv for suspicious stop frames.")
    parser.add_argument("folder", nargs="?", default=".",
                        help="Folder containing *_clips.csv files (default: current dir)")
    parser.add_argument("--threshold", type=int, default=THRESHOLD,
                        help=f"Frame distance to flag (default: {THRESHOLD})")
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    print(f"Scanning: {folder}  (threshold ±{args.threshold} frames)\n")
    scan_folder(folder, args.threshold)


if __name__ == "__main__":
    main()
