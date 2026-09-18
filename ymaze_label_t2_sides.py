#!/usr/bin/env python3
"""
Fast manual left/right annotation for Y-maze T2 clips, followed by safe insertion
of side-hit measures into an existing measure-long 2-day-group contract CSV.

Default paths are configured for Shuting's Y-maze project:
  Clips:
    E:\\ymaze\\t1t2_and_clips
  Contract:
    \\\\moorelaboratory.dts.usc.edu\\Shared\\Shuting\\P1-SNr\\Figures-P1-SNr\\Data\\y-maze\\ymaze_contract_hits_2daygroupStatistics.csv

What it does
------------
1. Recursively scans the clips folder and keeps T2 video clips only.
2. Matches T2 clips to the labeled trial log, preferably by ID + Day + start_frame.
   If start_frame is not encoded in the filenames, it falls back to within-mouse/day
   clip order ONLY when clip count exactly matches raw T2-row count.
3. Shows the final frame of each clip for rapid side labeling:
       L = screen-left arm
       R = screen-right arm
       U = unclear
       SPACE = play/replay clip
       B / Backspace = go back one clip
       Q / Esc = save and quit
   Labels autosave after every key press, so the program can be resumed safely.
4. Once every HIT T2 trial needed by the target contract has a known side, it creates:
       left_side_hits
       right_side_hits
   for each mouse and each 2-day group represented in the existing contract.
5. Makes a timestamped backup of the contract, removes any old rows for these two
   measures, appends the rebuilt rows, and writes the contract back in place.
6. Also writes an audit CSV with left/right choices, hits, and a side-bias index.

The annotation GUI intentionally does NOT display genotype or hit/miss status, reducing
observer bias while coding side choice.

Dependencies
------------
    pip install pandas numpy opencv-python

Run
---
    python ymaze_label_t2_sides.py

Optional arguments
------------------
    python ymaze_label_t2_sides.py --clips "E:\\ymaze\\t1t2_and_clips" \
        --contract "...\\ymaze_contract_hits_2daygroupStatistics.csv" \
        --raw-log "...\\ymaze_time_log_labeled v22June26.csv"

Use --dry-run to do matching/QC without opening the annotation GUI or modifying the contract.
Use --no-merge to annotate and save side labels but not modify the contract.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_CLIPS = Path(r"E:\ymaze\t1t2_and_clips")
DEFAULT_CONTRACT = Path(
    r"\\moorelaboratory.dts.usc.edu\Shared\Shuting\P1-SNr\Figures-P1-SNr\Data\y-maze\ymaze_contract_hits_2daygroupStatistics.csv"
)
DEFAULT_RAW_LOG_NAME = "ymaze_time_log_labeled v22June26.csv"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".wmv"}
SIDE_MEASURES = ("left_side_hits", "right_side_hits")
ALIASES = {"Ctrl": "Control", "ctrl": "Control", "control": "Control"}


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def norm_id(x: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())


def natural_key(s: object):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def is_t2_clip(path: Path, clips_root: Path) -> bool:
    """Avoid treating the root folder name 't1t2_and_clips' itself as a T2 marker."""
    stem = path.stem.lower()
    if re.search(r"(?:^|[_\-\s])t2(?:$|[_\-\s.])", stem):
        return True
    try:
        rel_parts = path.relative_to(clips_root).parts[:-1]
    except Exception:
        rel_parts = path.parts[:-1]
    for part in rel_parts:
        p = part.lower()
        if p == "t2" or re.match(r"^t2(?:[_\-\s]|$)", p):
            return True
    return False


def parse_mouse_id(path: Path, valid_ids: Dict[str, str]) -> Optional[str]:
    """Return exact target/raw ID spelling if a known ID appears in the path."""
    text = str(path)
    # First: direct normalized matching against known IDs, longest first.
    compact = norm_id(text)
    candidates = [exact for nrm, exact in valid_ids.items() if nrm and nrm in compact]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Prefer the longest normalized ID to avoid prefix collisions.
        candidates.sort(key=lambda x: len(norm_id(x)), reverse=True)
        if len(norm_id(candidates[0])) > len(norm_id(candidates[1])):
            return candidates[0]

    # Fallback regex for common IDs.
    m = re.search(r"(?i)(?:^|[^A-Za-z0-9])((?:SC|JC|LM)[_\- ]?\d{1,3})(?:[^A-Za-z0-9]|$)", text)
    if m:
        n = norm_id(m.group(1))
        return valid_ids.get(n, n)
    return None


def parse_day(path: Path) -> Optional[int]:
    """Recognize d1, day1, day_1, day-1 anywhere below the clip root."""
    text = str(path)
    pats = [
        r"(?i)(?:^|[^A-Za-z0-9])day[_\-\s]*0*(\d{1,2})(?:[^0-9]|$)",
        r"(?i)(?:^|[^A-Za-z0-9])d[_\-\s]*0*(\d{1,2})(?:[^0-9]|$)",
    ]
    hits = []
    for pat in pats:
        hits.extend(int(x) for x in re.findall(pat, text))
    hits = [x for x in hits if 0 <= x <= 99]
    return hits[-1] if hits else None


def parse_start_frame(path: Path) -> Optional[int]:
    """Conservative start-frame parser. Returns None rather than guessing from bare numbers."""
    text = path.stem
    pats = [
        r"(?i)(?:start[_\- ]*frame|startframe|start|sf)[_\- ]*(\d{2,9})",
        r"(?i)(?:^|[_\- ])frame[_\- ]*(\d{2,9})(?:[_\- ]|$)",
    ]
    for pat in pats:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return None


def find_raw_log(contract: Path, explicit: Optional[Path], clips_root: Optional[Path] = None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Raw labeled log not found: {explicit}")
        return explicit

    search_dirs = [contract.parent]
    if clips_root is not None:
        search_dirs.extend([clips_root, clips_root.parent])
    # Deduplicate directories while preserving priority.
    dirs = []
    seen_dirs = set()
    for d in search_dirs:
        k = str(d).lower()
        if k not in seen_dirs and d.is_dir():
            seen_dirs.add(k)
            dirs.append(d)

    for d in dirs:
        exact = d / DEFAULT_RAW_LOG_NAME
        if exact.is_file():
            return exact

    cands = []
    for d in dirs:
        cands += list(d.glob("*ymaze*time*log*labeled*.csv"))
        cands += list(d.glob("*ymaze*log*labeled*.csv"))
    # Deduplicate, avoid derived contracts/stats.
    uniq = []
    seen = set()
    for p in cands:
        rp = str(p.resolve()).lower()
        if rp not in seen and "contract" not in p.name.lower() and "stats" not in p.name.lower():
            seen.add(rp)
            uniq.append(p)
    if len(uniq) == 1:
        return uniq[0]
    if len(uniq) > 1:
        print("\nMultiple candidate labeled logs found:")
        for i, p in enumerate(uniq, 1):
            print(f"  {i}. {p}")
        print("Using the first candidate. Override with --raw-log if needed.")
        return sorted(uniq, key=lambda p: p.name.lower())[0]

    raise FileNotFoundError(
        "Could not find the labeled trial log beside the contract, in the clips folder, or in its parent.\n"
        "Pass it explicitly with --raw-log."
    )


# -----------------------------------------------------------------------------
# Target contract / 2-day bucket detection
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Bucket:
    label: object
    days: Tuple[int, ...]


def _days_from_label(x: object) -> Optional[Tuple[int, ...]]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    sl = s.lower()
    nums = [int(n) for n in re.findall(r"(?<!\d)([1-6])(?!\d)", s)]
    nums = list(dict.fromkeys(nums))
    if len(nums) >= 2:
        return tuple(nums[:2])
    if "early" in sl:
        return (1, 2)
    if "middle" in sl or re.search(r"\bmid\b", sl):
        return (3, 4)
    if "late" in sl:
        return (5, 6)
    return None


def detect_bucket_column(contract_df: pd.DataFrame, contract_path: Path) -> Tuple[str, List[Bucket]]:
    candidates = [
        "DayGroup", "Day_Group", "day_group", "DayPair", "Day_Pair", "TwoDayGroup",
        "Two_Day_Group", "Epoch", "Session", "Days", "Day"
    ]
    present = [c for c in candidates if c in contract_df.columns]
    if not present:
        raise ValueError(
            "Could not identify the 2-day-group column. Expected one of: " + ", ".join(candidates)
        )

    for col in present:
        vals = [v for v in contract_df[col].dropna().unique().tolist()]
        buckets: List[Bucket] = []

        # Text labels such as 'Early (d1-2)', 'Days 3-4', 'Late (d5-6)'.
        for v in vals:
            ds = _days_from_label(v)
            if ds:
                buckets.append(Bucket(v, ds))

        # If all/most unique values parsed, use this column.
        if buckets and len({str(b.label) for b in buckets}) >= min(2, len(vals)):
            # Deduplicate by exact label string.
            seen = set()
            out = []
            for b in buckets:
                k = str(b.label)
                if k not in seen:
                    seen.add(k)
                    out.append(b)
            return col, out

        # Special case: numeric group index 1,2,3 in a file explicitly named 2daygroup.
        if col == "Day" and "2daygroup" in contract_path.name.lower():
            numeric = pd.to_numeric(pd.Series(vals), errors="coerce")
            if numeric.notna().all():
                ints = sorted(set(int(x) for x in numeric))
                if set(ints).issubset({1, 2, 3}) and len(ints) >= 2:
                    mp = {1: (1, 2), 2: (3, 4), 3: (5, 6)}
                    return col, [Bucket(i, mp[i]) for i in ints]

    raise ValueError(
        "Found possible grouping columns but could not map their labels to 2-day windows.\n"
        + "\n".join(f"  {c}: {contract_df[c].dropna().unique()[:10].tolist()}" for c in present)
    )


def validate_contract(contract_df: pd.DataFrame):
    required = {"ID", "Group", "Measure", "Value"}
    missing = required - set(contract_df.columns)
    if missing:
        raise ValueError(
            f"Target contract is missing required columns {sorted(missing)}.\n"
            f"Columns found: {contract_df.columns.tolist()}"
        )


# -----------------------------------------------------------------------------
# Clip inventory and matching
# -----------------------------------------------------------------------------
def build_clip_inventory(clips_root: Path, valid_ids: Dict[str, str]) -> pd.DataFrame:
    vids = sorted(
        [p for p in clips_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS and is_t2_clip(p, clips_root)],
        key=lambda p: natural_key(str(p.relative_to(clips_root)))
    )
    rows = []
    for p in vids:
        rows.append({
            "clip_path": str(p),
            "clip_rel": str(p.relative_to(clips_root)),
            "ID": parse_mouse_id(p, valid_ids),
            "Day": parse_day(p),
            "start_frame_from_name": parse_start_frame(p),
        })
    return pd.DataFrame(rows)


def prepare_raw_t2(raw_log: Path, valid_ids: Dict[str, str]) -> pd.DataFrame:
    raw = pd.read_csv(raw_log)
    needed = {"ID", "Day", "time_point", "correct"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"Raw labeled log is missing columns: {sorted(missing)}")

    raw = raw.copy()
    raw["_IDnorm"] = raw["ID"].map(norm_id)
    raw = raw[raw["_IDnorm"].isin(valid_ids.keys())]
    raw["ID"] = raw["_IDnorm"].map(valid_ids)
    raw["Day"] = pd.to_numeric(raw["Day"], errors="coerce")
    raw["correct"] = pd.to_numeric(raw["correct"], errors="coerce")
    raw = raw[raw["time_point"].astype(str).str.lower().eq("t2")]

    if "start_frame" not in raw.columns:
        raw["start_frame"] = np.nan
    raw["start_frame"] = pd.to_numeric(raw["start_frame"], errors="coerce")
    raw = raw.sort_values(["ID", "Day", "start_frame"], kind="stable").reset_index(drop=False)
    raw = raw.rename(columns={"index": "raw_source_index"})
    raw["raw_t2_uid"] = np.arange(len(raw), dtype=int)
    return raw


def match_clips_to_raw(inv: pd.DataFrame, raw_t2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inv = inv.copy()
    for c in ["raw_t2_uid", "raw_source_index", "start_frame", "correct", "match_method"]:
        inv[c] = np.nan if c != "match_method" else ""

    raw_used = set()

    # Pass 1: exact ID + Day + start_frame.
    for i, r in inv.iterrows():
        if pd.isna(r.get("ID")) or pd.isna(r.get("Day")) or pd.isna(r.get("start_frame_from_name")):
            continue
        cand = raw_t2[
            (raw_t2["ID"] == r["ID"]) &
            (raw_t2["Day"] == float(r["Day"])) &
            (raw_t2["start_frame"] == float(r["start_frame_from_name"])) &
            (~raw_t2["raw_t2_uid"].isin(raw_used))
        ]
        if len(cand) == 1:
            rr = cand.iloc[0]
            uid = int(rr["raw_t2_uid"])
            raw_used.add(uid)
            inv.loc[i, ["raw_t2_uid", "raw_source_index", "start_frame", "correct", "match_method"]] = [
                uid, rr["raw_source_index"], rr["start_frame"], rr["correct"], "exact_start_frame"
            ]

    # Pass 2: within ID/day order only when remaining counts match exactly.
    group_keys = inv.dropna(subset=["ID", "Day"])[["ID", "Day"]].drop_duplicates().itertuples(index=False, name=None)
    for mid, day in group_keys:
        clip_idx = inv.index[
            (inv["ID"] == mid) & (inv["Day"] == day) & inv["raw_t2_uid"].isna()
        ].tolist()
        if not clip_idx:
            continue
        raw_rem = raw_t2[
            (raw_t2["ID"] == mid) &
            (raw_t2["Day"] == float(day)) &
            (~raw_t2["raw_t2_uid"].isin(raw_used))
        ].sort_values("start_frame", kind="stable")
        if len(clip_idx) != len(raw_rem):
            continue
        clip_idx = sorted(clip_idx, key=lambda idx: natural_key(inv.at[idx, "clip_rel"]))
        for idx, (_, rr) in zip(clip_idx, raw_rem.iterrows()):
            uid = int(rr["raw_t2_uid"])
            raw_used.add(uid)
            inv.loc[idx, ["raw_t2_uid", "raw_source_index", "start_frame", "correct", "match_method"]] = [
                uid, rr["raw_source_index"], rr["start_frame"], rr["correct"], "ordered_count_exact"
            ]

    unmatched_raw = raw_t2[~raw_t2["raw_t2_uid"].isin(raw_used)].copy()
    return inv, unmatched_raw


# -----------------------------------------------------------------------------
# Label storage and OpenCV annotation UI
# -----------------------------------------------------------------------------
def load_existing_labels(labels_path: Path) -> Dict[str, str]:
    if not labels_path.is_file():
        return {}
    try:
        old = pd.read_csv(labels_path)
    except Exception:
        return {}
    if not {"clip_path", "side"}.issubset(old.columns):
        return {}
    out = {}
    for _, r in old.iterrows():
        s = str(r["side"]).upper().strip()
        if s in {"L", "R", "U"}:
            out[str(r["clip_path"])] = s
    return out


def save_labels(inv: pd.DataFrame, labels_path: Path):
    cols = [
        "clip_path", "clip_rel", "ID", "Day", "start_frame_from_name",
        "start_frame", "raw_source_index", "raw_t2_uid", "match_method", "side"
    ]
    # Deliberately omit 'correct' from the human-facing label file? Keep it for auditability,
    # but the GUI does not show it.
    out = inv.copy()
    if "correct" in out.columns:
        cols.append("correct")
    out[cols].to_csv(labels_path, index=False)


def read_last_frame(path: Path) -> Tuple[Optional[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 30.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame = None
    if n > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 2))
        ok, fr = cap.read()
        if ok:
            frame = fr
    if frame is None:
        # Reliable fallback for codecs with poor random seeking.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frame = fr
    cap.release()
    return frame, float(fps)


def fit_frame(frame: np.ndarray, max_w: int = 1280, max_h: int = 850) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale < 1.0:
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame.copy()


def add_overlay(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.68
    thick = 2
    pad = 10
    line_h = 30
    box_h = pad * 2 + line_h * len(lines)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], min(box_h, out.shape[0])), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, out, 0.32, 0, out)
    y = pad + 22
    for line in lines:
        cv2.putText(out, line, (12, y), font, fs, (255, 255, 255), thick, cv2.LINE_AA)
        y += line_h
    return out


def play_clip(path: Path, window: str):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    delay = max(1, int(round(1000.0 / fps)))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        show = fit_frame(frame)
        show = add_overlay(show, ["PLAYBACK - SPACE/ESC to stop"])
        cv2.imshow(window, show)
        key = cv2.waitKey(delay) & 0xFF
        if key in (27, 32):
            break
    cap.release()


def annotate(inv: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    inv = inv.copy()
    existing = load_existing_labels(labels_path)
    inv["side"] = inv["clip_path"].map(existing).fillna("")

    # Only matched clips are useful for the contract; unmatched remain in QC inventory.
    eligible = inv.index[inv["raw_t2_uid"].notna()].tolist()
    if not eligible:
        print("No matched T2 clips available for annotation.")
        save_labels(inv, labels_path)
        return inv

    # Resume at first unlabeled/unclear clip; U is considered unresolved and shown again.
    pos = 0
    for j, idx in enumerate(eligible):
        if str(inv.at[idx, "side"]).upper() not in {"L", "R"}:
            pos = j
            break
    else:
        print("All matched clips already have L/R labels.")
        save_labels(inv, labels_path)
        return inv

    window = "Y-maze T2 side label | L/R | Space play | B back | U unclear | Q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while 0 <= pos < len(eligible):
        idx = eligible[pos]
        r = inv.loc[idx]
        path = Path(r["clip_path"])
        frame, _ = read_last_frame(path)
        if frame is None:
            print(f"Could not read: {path}")
            inv.at[idx, "side"] = "U"
            save_labels(inv, labels_path)
            pos += 1
            continue

        show = fit_frame(frame)
        prev = str(inv.at[idx, "side"]).upper() or "unlabeled"
        lines = [
            f"{pos+1}/{len(eligible)}   ID={r['ID']}   Day={int(r['Day']) if pd.notna(r['Day']) else '?'}",
            f"{r['clip_rel']}",
            f"Current={prev}   L=screen-left   R=screen-right   U=unclear   SPACE=play   B=back   Q=save+quit",
        ]
        show = add_overlay(show, lines)
        cv2.imshow(window, show)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("l"), ord("L")):
            inv.at[idx, "side"] = "L"
            save_labels(inv, labels_path)
            pos += 1
        elif key in (ord("r"), ord("R")):
            inv.at[idx, "side"] = "R"
            save_labels(inv, labels_path)
            pos += 1
        elif key in (ord("u"), ord("U")):
            inv.at[idx, "side"] = "U"
            save_labels(inv, labels_path)
            pos += 1
        elif key in (ord("b"), ord("B"), 8):
            pos = max(0, pos - 1)
        elif key == 32:  # space
            play_clip(path, window)
        elif key in (ord("q"), ord("Q"), 27):
            save_labels(inv, labels_path)
            break
        else:
            # Ignore unknown keys.
            continue

    cv2.destroyAllWindows()
    save_labels(inv, labels_path)
    return inv


# -----------------------------------------------------------------------------
# Aggregation / merge
# -----------------------------------------------------------------------------
def matched_annotation_table(inv: pd.DataFrame, raw_t2: pd.DataFrame) -> pd.DataFrame:
    m = inv[inv["raw_t2_uid"].notna()].copy()
    m["raw_t2_uid"] = pd.to_numeric(m["raw_t2_uid"], errors="coerce").astype("Int64")
    keep = raw_t2[["raw_t2_uid", "ID", "Day", "correct", "start_frame"]].copy()
    keep["raw_t2_uid"] = keep["raw_t2_uid"].astype("Int64")
    out = m.drop(columns=[c for c in ["ID", "Day", "correct", "start_frame"] if c in m.columns]).merge(
        keep, on="raw_t2_uid", how="left", validate="many_to_one"
    )
    out["side"] = out["side"].astype(str).str.upper().str.strip()
    return out


def relevant_days(buckets: Sequence[Bucket]) -> set:
    return {d for b in buckets for d in b.days}


def qc_before_merge(ann: pd.DataFrame, raw_t2: pd.DataFrame, buckets: Sequence[Bucket], valid_ids: Dict[str, str]) -> Tuple[bool, List[str]]:
    msgs = []
    days = relevant_days(buckets)
    target_ids = set(valid_ids.values())

    raw_rel = raw_t2[raw_t2["ID"].isin(target_ids) & raw_t2["Day"].isin(days)].copy()
    ann_rel = ann[ann["ID"].isin(target_ids) & ann["Day"].isin(days)].copy()

    matched_uids = set(pd.to_numeric(ann_rel["raw_t2_uid"], errors="coerce").dropna().astype(int))
    raw_hits = raw_rel[pd.to_numeric(raw_rel["correct"], errors="coerce").eq(1)]
    missing_hit_raw = raw_hits[~raw_hits["raw_t2_uid"].isin(matched_uids)]
    unresolved_hit = ann_rel[
        pd.to_numeric(ann_rel["correct"], errors="coerce").eq(1) & ~ann_rel["side"].isin(["L", "R"])
    ]

    if len(missing_hit_raw):
        msgs.append(f"BLOCK: {len(missing_hit_raw)} hit T2 rows in the raw log have no matched video clip.")
    if len(unresolved_hit):
        msgs.append(f"BLOCK: {len(unresolved_hit)} matched hit T2 clips are still unlabeled/unclear.")

    unresolved_all = ann_rel[~ann_rel["side"].isin(["L", "R"])]
    if len(unresolved_all):
        msgs.append(
            f"NOTE: {len(unresolved_all)} total T2 clips are unresolved; side-hit counts are still valid if none are hits, "
            "but all-choice side-bias metrics will be incomplete."
        )

    return (len(missing_hit_raw) == 0 and len(unresolved_hit) == 0), msgs


def make_side_audit(ann: pd.DataFrame, contract_df: pd.DataFrame, bucket_col: str, buckets: Sequence[Bucket]) -> pd.DataFrame:
    # Exact target ID/group pairs from contract.
    id_group = contract_df[["ID", "Group"]].drop_duplicates().copy()
    rows = []
    for _, ig in id_group.iterrows():
        mid, grp = ig["ID"], ig["Group"]
        for b in buckets:
            s = ann[(ann["ID"] == mid) & ann["Day"].isin(b.days) & ann["side"].isin(["L", "R"])].copy()
            left_choices = int((s["side"] == "L").sum())
            right_choices = int((s["side"] == "R").sum())
            hits = pd.to_numeric(s["correct"], errors="coerce").eq(1)
            left_hits = int(((s["side"] == "L") & hits).sum())
            right_hits = int(((s["side"] == "R") & hits).sum())
            total_choices = left_choices + right_choices
            total_hits = left_hits + right_hits
            bias = (right_choices - left_choices) / total_choices if total_choices else np.nan
            rows.append({
                "ID": mid,
                "Group": grp,
                bucket_col: b.label,
                "Days": "+".join(str(d) for d in b.days),
                "left_choices": left_choices,
                "right_choices": right_choices,
                "left_side_hits": left_hits,
                "right_side_hits": right_hits,
                "total_choices_with_known_side": total_choices,
                "total_hits_with_known_side": total_hits,
                "side_bias_index_RminusL_over_total": bias,
            })
    return pd.DataFrame(rows)


def make_contract_side_rows(contract_df: pd.DataFrame, bucket_col: str, buckets: Sequence[Bucket], audit: pd.DataFrame) -> pd.DataFrame:
    # Only add ID x bucket combinations that already exist in the contract.
    key_existing = contract_df[["ID", "Group", bucket_col]].drop_duplicates().copy()
    bucket_day_map = {str(b.label): b for b in buckets}

    out_rows = []
    for _, k in key_existing.iterrows():
        label = k[bucket_col]
        b = bucket_day_map.get(str(label))
        if b is None:
            continue
        a = audit[
            (audit["ID"] == k["ID"]) &
            (audit["Group"] == k["Group"]) &
            (audit[bucket_col].astype(str) == str(label))
        ]
        if len(a) != 1:
            continue
        ar = a.iloc[0]
        for meas in SIDE_MEASURES:
            row = {c: np.nan for c in contract_df.columns}
            row["ID"] = k["ID"]
            row["Group"] = k["Group"]
            row[bucket_col] = label
            row["Measure"] = meas
            row["Value"] = int(ar[meas])
            # Existing contract convention: n_trials is number of hit trials underlying the side split.
            if "n_trials" in contract_df.columns:
                row["n_trials"] = int(ar["total_hits_with_known_side"])
            out_rows.append(row)
    return pd.DataFrame(out_rows, columns=contract_df.columns)


def merge_into_contract(contract_path: Path, contract_df: pd.DataFrame, side_rows: pd.DataFrame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = contract_path.with_name(contract_path.stem + f"_backup_before_sidehits_{ts}" + contract_path.suffix)
    shutil.copy2(contract_path, backup)

    old_n = len(contract_df)
    meas_lower = contract_df["Measure"].astype(str).str.lower()
    keep = ~meas_lower.isin({m.lower() for m in SIDE_MEASURES})
    cleaned = contract_df[keep].copy()
    final = pd.concat([cleaned, side_rows], ignore_index=True)
    final.to_csv(contract_path, index=False)

    print(f"\nContract updated in place: {contract_path}")
    print(f"Backup:                   {backup}")
    print(f"Rows: {old_n} -> {len(final)}  (appended {len(side_rows)} rebuilt side-hit rows)")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    ap.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    ap.add_argument("--raw-log", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="match/QC only; do not annotate or merge")
    ap.add_argument("--no-merge", action="store_true", help="annotate and save labels, but do not modify contract")
    args = ap.parse_args()

    clips_root: Path = args.clips
    contract_path: Path = args.contract

    if not clips_root.is_dir():
        sys.exit(f"Clips folder not found: {clips_root}")
    if not contract_path.is_file():
        sys.exit(f"Target contract not found: {contract_path}")

    contract_df = pd.read_csv(contract_path)
    validate_contract(contract_df)
    bucket_col, buckets = detect_bucket_column(contract_df, contract_path)
    print(f"2-day grouping column: {bucket_col}")
    for b in buckets:
        print(f"  {b.label!r} -> days {b.days}")

    # Restrict to IDs already represented in the target contract. This automatically follows
    # the current inclusion/exclusion set rather than hard-coding another mouse list.
    exact_ids = contract_df["ID"].dropna().astype(str).unique().tolist()
    valid_ids = {norm_id(x): x for x in exact_ids}
    print(f"Target contract mice: {len(exact_ids)}")

    raw_log = find_raw_log(contract_path, args.raw_log, clips_root)
    print(f"Raw labeled log: {raw_log}")

    inv = build_clip_inventory(clips_root, valid_ids)
    if inv.empty:
        sys.exit(
            "No T2 video clips were found. A T2 clip must have 't2' in its filename "
            "or be inside a subfolder named T2."
        )
    print(f"T2 video clips found: {len(inv)}")
    print(f"  ID parsed:  {inv['ID'].notna().sum()}/{len(inv)}")
    print(f"  Day parsed: {inv['Day'].notna().sum()}/{len(inv)}")
    print(f"  start_frame parsed from filename: {inv['start_frame_from_name'].notna().sum()}/{len(inv)}")

    raw_t2 = prepare_raw_t2(raw_log, valid_ids)
    print(f"Raw labeled T2 rows for target mice: {len(raw_t2)}")

    inv, unmatched_raw = match_clips_to_raw(inv, raw_t2)
    n_exact = int((inv["match_method"] == "exact_start_frame").sum())
    n_order = int((inv["match_method"] == "ordered_count_exact").sum())
    n_unmatched_clips = int(inv["raw_t2_uid"].isna().sum())
    print("\nMatching summary:")
    print(f"  exact start-frame matches : {n_exact}")
    print(f"  safe order matches        : {n_order}")
    print(f"  unmatched clips           : {n_unmatched_clips}")
    print(f"  unmatched raw T2 rows     : {len(unmatched_raw)}")

    labels_path = clips_root / "ymaze_side_labels.csv"
    inventory_path = clips_root / "ymaze_t2_clip_inventory_QC.csv"
    inv.to_csv(inventory_path, index=False)
    print(f"QC inventory: {inventory_path}")

    # Helpful unmatched diagnostics.
    if n_unmatched_clips:
        print("\nFirst unmatched clips:")
        print(inv[inv["raw_t2_uid"].isna()][["clip_rel", "ID", "Day", "start_frame_from_name"]].head(15).to_string(index=False))
    if len(unmatched_raw):
        print("\nFirst unmatched raw T2 rows:")
        print(unmatched_raw[["ID", "Day", "start_frame", "correct"]].head(15).to_string(index=False))

    if args.dry_run:
        print("\nDry run complete. No labels or contract changes were made.")
        return

    inv = annotate(inv, labels_path)
    ann = matched_annotation_table(inv, raw_t2)

    okay, qc_msgs = qc_before_merge(ann, raw_t2, buckets, valid_ids)
    print("\nPre-merge QC:")
    if qc_msgs:
        for msg in qc_msgs:
            print("  " + msg)
    else:
        print("  PASS: all required hit T2 trials are matched and have L/R labels.")

    audit = make_side_audit(ann, contract_df, bucket_col, buckets)
    audit_path = contract_path.parent / "ymaze_side_choice_2daygroup_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"Side-choice audit: {audit_path}")

    side_rows = make_contract_side_rows(contract_df, bucket_col, buckets, audit)
    preview_path = contract_path.parent / "ymaze_side_hit_contract_rows_preview.csv"
    side_rows.to_csv(preview_path, index=False)
    print(f"Rows prepared for contract: {preview_path}")

    if args.no_merge:
        print("\n--no-merge selected. Contract was not modified.")
        return

    if not okay:
        print(
            "\nCONTRACT NOT MODIFIED because required hit-side information is incomplete.\n"
            "Fix the unmatched clips or label the unresolved HIT clips, then rerun.\n"
            "Your existing labels are already saved, so you will resume rather than start over."
        )
        return

    merge_into_contract(contract_path, contract_df, side_rows)

    # Concise final side-hit check.
    print("\nSide-hit totals by group and 2-day bucket:")
    if not audit.empty:
        g = audit.groupby([bucket_col, "Group"], dropna=False)[["left_side_hits", "right_side_hits"]].sum()
        print(g.to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
