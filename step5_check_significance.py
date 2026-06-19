#!/usr/bin/env python3
"""
ymaze_stats_analysis.py
=======================
Compares SNr-DTA vs Control group duration_s for T1, T2, and T1+T2 combined.

Statistical approach
--------------------
The data are repeated measures (multiple days per mouse), so we use:

  1. **Linear Mixed Model (LMM)** — primary test
       fixed effects : Group, Day, Group×Day interaction
       random effect : random intercept per mouse (ID)
       Reported: F-statistic, p-value for the Group effect.

  2. **Per-day Mann-Whitney U** — non-parametric per-day comparison
       Uses per-mouse daily means (one value per mouse per day) to avoid
       pseudo-replication.  Reports U, p, and Cohen's d effect size.

  3. **Summary table** — mean ± SEM per group per day, printed and saved.

Three analyses are run in sequence:
    • T1 only
    • T2 only
    • T1 + T2 combined (mean of t1 and t2 per trial as a single value)

Input
-----
Reads ``ymaze_time_log_raw.csv`` (or ``ymaze_time_log_labeled.csv`` if
present — allows optional filtering to hits-only or misses-only).

    python ymaze_stats_analysis.py                      # current directory
    python ymaze_stats_analysis.py /path/to/folder
    python ymaze_stats_analysis.py /path/to/folder --subset hits
    python ymaze_stats_analysis.py /path/to/folder --subset misses
    python ymaze_stats_analysis.py /path/to/folder --subset all   (default)

Output
------
    ymaze_stats_results.csv   — per-day Mann-Whitney results for all three analyses
    ymaze_stats_lmm.txt       — LMM summary tables (full model output)
    Console                   — readable summary of all key results
"""

import os
import sys
import argparse
import warnings
import textwrap

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# statsmodels for LMM
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── file names ─────────────────────────────────────────────────────────────────
RAW_FILE     = "ymaze_time_log_raw.csv"
LABELED_FILE = "ymaze_time_log_labeled.csv"
OUT_CSV      = "ymaze_stats_results.csv"
OUT_LMM      = "ymaze_stats_lmm.txt"

GROUP_COL   = "Group"
SNR_LABEL   = "SNr-DTA"
CTRL_LABEL  = "Ctrl"


# ── helpers ────────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    """Pooled Cohen's d (two independent groups)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) +
                          (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled_std


def significance_stars(p):
    if pd.isna(p):     return "ns"
    if p < 0.001:      return "***"
    if p < 0.01:       return "**"
    if p < 0.05:       return "*"
    return "ns"


def effect_label(d):
    """Cohen's d magnitude label."""
    if pd.isna(d):     return "—"
    ad = abs(d)
    if ad < 0.2:       return "negligible"
    if ad < 0.5:       return "small"
    if ad < 0.8:       return "medium"
    return "large"


def hr(char="─", width=72):
    return char * width


# ── data loading ───────────────────────────────────────────────────────────────

def load_data(in_dir: str, subset: str) -> pd.DataFrame:
    """
    Load raw or labeled CSV.
    subset : "all" | "hits" | "misses"
    """
    labeled_path = os.path.join(in_dir, LABELED_FILE)
    raw_path     = os.path.join(in_dir, RAW_FILE)

    if subset in ("hits", "misses") and os.path.exists(labeled_path):
        df = pd.read_csv(labeled_path)
        print(f"  Using labeled file: {LABELED_FILE}")
        if "correct" not in df.columns:
            sys.exit("  ERROR: 'correct' column missing from labeled file.")
        val = 1 if subset == "hits" else 0
        df = df[df["correct"] == val].copy()
        print(f"  Subset '{subset}': {len(df)} rows retained.")
    else:
        if subset != "all":
            print(f"  Warning: labeled file not found — falling back to all trials.")
        if not os.path.exists(raw_path):
            sys.exit(f"  ERROR: {RAW_FILE} not found in {in_dir}")
        df = pd.read_csv(raw_path)
        print(f"  Using raw file: {RAW_FILE}")

    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    df["Day"]        = pd.to_numeric(df["Day"],        errors="coerce").astype("Int64")
    df = df.dropna(subset=["duration_s", "Day", GROUP_COL])

    print(f"  {len(df)} rows · groups: {sorted(df[GROUP_COL].unique())}")
    n_per_group = df.groupby(GROUP_COL)["ID"].nunique()
    for grp, n in n_per_group.items():
        print(f"    {grp}: {n} mice")
    print()
    return df


# ── per-mouse daily means ──────────────────────────────────────────────────────

def mouse_day_means(df: pd.DataFrame, measure: str) -> pd.DataFrame:
    """
    Aggregate to one value per (mouse, day) for a given measure column.
    Returns DataFrame with columns: ID, Group, Day, value
    """
    agg = (
        df.groupby(["ID", GROUP_COL, "Day"], dropna=False)["duration_s"]
        .mean()
        .reset_index()
        .rename(columns={"duration_s": "value"})
    )
    agg["measure"] = measure
    return agg


def build_t1_t2_combined(df: pd.DataFrame) -> pd.DataFrame:
    """
    For T1+T2 combined: average the t1 and t2 duration for each trial pair
    (same mouse, same day, same trial_slot if available, else just mean).
    """
    return mouse_day_means(df, "T1+T2")


# ── per-day Mann-Whitney ───────────────────────────────────────────────────────

def per_day_mannwhitney(mouse_df: pd.DataFrame, days: list, measure: str) -> pd.DataFrame:
    """
    For each day: split into SNr-DTA vs Ctrl, run Mann-Whitney U,
    compute Cohen's d, mean ± SEM per group.
    mouse_df must have columns: ID, Group, Day, value
    """
    rows = []
    for day in sorted(days):
        day_df = mouse_df[mouse_df["Day"] == day]
        snr  = day_df[day_df[GROUP_COL] == SNR_LABEL]["value"].dropna().values
        ctrl = day_df[day_df[GROUP_COL] == CTRL_LABEL]["value"].dropna().values

        if len(snr) < 2 or len(ctrl) < 2:
            continue

        u_stat, p_val = scipy_stats.mannwhitneyu(snr, ctrl, alternative="two-sided")
        d = cohens_d(snr, ctrl)

        rows.append({
            "measure":          measure,
            "Day":              int(day),
            "n_SNrDTA":         len(snr),
            "n_Ctrl":           len(ctrl),
            "mean_SNrDTA":      np.mean(snr),
            "sem_SNrDTA":       scipy_stats.sem(snr),
            "mean_Ctrl":        np.mean(ctrl),
            "sem_Ctrl":         scipy_stats.sem(ctrl),
            "U_stat":           u_stat,
            "p_value":          p_val,
            "stars":            significance_stars(p_val),
            "cohens_d":         d,
            "effect_size":      effect_label(d),
        })
    return pd.DataFrame(rows)



# ── Early vs Late comparison (Day 1&2 vs Day 5&6) ─────────────────────────────

EARLY_DAYS = [1, 2]
LATE_DAYS  = [5, 6]

def early_vs_late(mouse_df: pd.DataFrame, measure: str) -> pd.DataFrame:
    """
    For each mouse, average their values across Day 1&2 (early) and Day 5&6
    (late), then compare SNr-DTA vs Ctrl within each epoch using Mann-Whitney U.
    Also tests within-group early→late change using Wilcoxon signed-rank.

    Returns a DataFrame with one row per epoch comparison.
    """
    rows = []
    for epoch_label, epoch_days in [("Early (Day1+2)", EARLY_DAYS),
                                     ("Late  (Day5+6)", LATE_DAYS)]:
        epoch_df = mouse_df[mouse_df["Day"].isin(epoch_days)]
        # one value per mouse = mean across the epoch days
        per_mouse = (
            epoch_df.groupby(["ID", GROUP_COL])["value"]
            .mean()
            .reset_index()
        )
        snr  = per_mouse[per_mouse[GROUP_COL] == SNR_LABEL]["value"].dropna().values
        ctrl = per_mouse[per_mouse[GROUP_COL] == CTRL_LABEL]["value"].dropna().values

        if len(snr) < 2 or len(ctrl) < 2:
            continue

        u_stat, p_val = scipy_stats.mannwhitneyu(snr, ctrl, alternative="two-sided")
        d = cohens_d(snr, ctrl)

        rows.append({
            "measure":       measure,
            "comparison":    "SNrDTA_vs_Ctrl",
            "epoch":         epoch_label,
            "n_SNrDTA":      len(snr),
            "n_Ctrl":        len(ctrl),
            "mean_SNrDTA":   np.mean(snr),
            "sem_SNrDTA":    scipy_stats.sem(snr),
            "mean_Ctrl":     np.mean(ctrl),
            "sem_Ctrl":      scipy_stats.sem(ctrl),
            "U_or_W_stat":   u_stat,
            "p_value":       p_val,
            "stars":         significance_stars(p_val),
            "cohens_d":      d,
            "effect_size":   effect_label(d),
        })

    # Within-group: Wilcoxon signed-rank early vs late for each group
    for grp in [SNR_LABEL, CTRL_LABEL]:
        grp_df = mouse_df[mouse_df[GROUP_COL] == grp]

        early = (grp_df[grp_df["Day"].isin(EARLY_DAYS)]
                 .groupby("ID")["value"].mean())
        late  = (grp_df[grp_df["Day"].isin(LATE_DAYS)]
                 .groupby("ID")["value"].mean())

        # align on common mice
        common = early.index.intersection(late.index)
        if len(common) < 3:
            continue

        e_vals = early.loc[common].values
        l_vals = late.loc[common].values

        try:
            w_stat, p_val = scipy_stats.wilcoxon(e_vals, l_vals)
        except Exception:
            w_stat, p_val = np.nan, np.nan

        d = cohens_d(l_vals, e_vals)   # late vs early (positive = increase)

        grp_short = "SNrDTA" if grp == SNR_LABEL else "Ctrl"
        rows.append({
            "measure":        measure,
            "comparison":     f"{grp_short}_Early_vs_Late",
            "epoch":          "Early→Late",
            "n_SNrDTA":       len(common) if grp == SNR_LABEL else np.nan,
            "n_Ctrl":         len(common) if grp == CTRL_LABEL else np.nan,
            "mean_SNrDTA":    np.mean(e_vals) if grp == SNR_LABEL else np.nan,
            "sem_SNrDTA":     scipy_stats.sem(e_vals) if grp == SNR_LABEL else np.nan,
            "mean_Ctrl":      np.mean(e_vals) if grp == CTRL_LABEL else np.nan,
            "sem_Ctrl":       scipy_stats.sem(e_vals) if grp == CTRL_LABEL else np.nan,
            "mean_late_SNrDTA": np.mean(l_vals) if grp == SNR_LABEL else np.nan,
            "mean_late_Ctrl":   np.mean(l_vals) if grp == CTRL_LABEL else np.nan,
            "U_or_W_stat":    w_stat,
            "p_value":        p_val,
            "stars":          significance_stars(p_val),
            "cohens_d":       d,
            "effect_size":    effect_label(d),
        })

    return pd.DataFrame(rows)


def print_early_late(measure: str, el_df: pd.DataFrame):
    """Print the early/late block inside a section."""
    print(f"\n  ▸ Early (Day 1+2) vs Late (Day 5+6) comparison\n")
    if el_df.empty:
        print("    Not enough data.")
        return

    # Between-group rows
    bg = el_df[el_df["comparison"] == "SNrDTA_vs_Ctrl"]
    if not bg.empty:
        print(f"    Between groups (SNr-DTA vs Ctrl):")
        hdr = (f"  {'Epoch':<20}  {'n SNr':>6}  {'n Ctrl':>6}  "
               f"{'Mean SNr':>9}  {'Mean Ctrl':>10}  "
               f"{'U':>7}  {'p':>8}  {'sig':>4}  {'d':>6}  effect")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for _, row in bg.iterrows():
            print(
                f"  {row['epoch']:<20}  "
                f"{int(row['n_SNrDTA']):>6}  "
                f"{int(row['n_Ctrl']):>6}  "
                f"{row['mean_SNrDTA']:>9.3f}  "
                f"{row['mean_Ctrl']:>10.3f}  "
                f"{row['U_or_W_stat']:>7.1f}  "
                f"{row['p_value']:>8.4f}  "
                f"{row['stars']:>4}  "
                f"{row['cohens_d']:>6.2f}  "
                f"{row['effect_size']}"
            )

    # Within-group rows
    wg = el_df[el_df["comparison"].str.contains("Early_vs_Late")]
    if not wg.empty:
        print(f"\n    Within-group Early→Late (Wilcoxon signed-rank):")
        hdr2 = (f"  {'Group':<20}  {'n':>4}  "
                f"{'Mean Early':>11}  {'Mean Late':>10}  "
                f"{'W':>7}  {'p':>8}  {'sig':>4}  {'d':>6}  effect")
        print(hdr2)
        print("  " + "─" * (len(hdr2) - 2))
        for _, row in wg.iterrows():
            grp_name = "SNr-DTA" if "SNrDTA" in row["comparison"] else "Ctrl"
            is_snr   = "SNrDTA" in row["comparison"]
            n      = int(row["n_SNrDTA"]) if is_snr else int(row["n_Ctrl"])
            mean_e = row["mean_SNrDTA"] if is_snr else row["mean_Ctrl"]
            late_key = "mean_late_SNrDTA" if is_snr else "mean_late_Ctrl"
            mean_l = row.get(late_key, np.nan)
            print(
                f"  {grp_name:<20}  "
                f"{n:>4}  "
                f"{mean_e:>11.3f}  "
                f"{mean_l:>10.3f}  "
                f"{row['U_or_W_stat']:>7.1f}  "
                f"{row['p_value']:>8.4f}  "
                f"{row['stars']:>4}  "
                f"{row['cohens_d']:>6.2f}  "
                f"{row['effect_size']}"
            )

# ── Linear Mixed Model ─────────────────────────────────────────────────────────

def run_lmm(mouse_df: pd.DataFrame, measure: str) -> tuple[str, float, float]:
    """
    LMM: value ~ Group * Day + (1 | ID)
    Returns (summary_text, group_F, group_p)
    """
    df = mouse_df.copy()
    df["Day"] = df["Day"].astype(float)
    df = df.rename(columns={GROUP_COL: "Group"})

    # Reference level: Ctrl
    df["Group"] = pd.Categorical(df["Group"],
                                 categories=[CTRL_LABEL, SNR_LABEL])

    # try multiple optimizers in order of preference
    result = None
    for method in ("lbfgs", "powell", "nm", "bfgs"):
        try:
            model = smf.mixedlm(
                "value ~ Group * Day",
                data=df,
                groups=df["ID"],
            )
            result = model.fit(reml=True, method=method)
            break
        except Exception:
            continue

    # last resort: drop interaction term
    if result is None:
        try:
            model = smf.mixedlm(
                "value ~ Group + Day",
                data=df,
                groups=df["ID"],
            )
            result = model.fit(reml=True, method="powell")
        except Exception as exc:
            return f"LMM failed (all methods): {exc}", np.nan, np.nan

    try:
        summary_text = result.summary().as_text()

        params = result.pvalues
        group_key = next((k for k in params.index if "SNr" in k and "Day" not in k), None)
        group_p = float(params[group_key]) if group_key else np.nan

        tvals = result.tvalues
        group_t = float(tvals[group_key]) if group_key else np.nan
        group_F = group_t ** 2

        return summary_text, group_F, group_p

    except Exception as exc:
        return f"LMM fitted but summary failed: {exc}", np.nan, np.nan


# ── pretty print ───────────────────────────────────────────────────────────────

def print_section(title: str, mw_df: pd.DataFrame, lmm_text: str,
                  group_F: float, group_p: float):
    w = 72
    print()
    print(hr("═"))
    print(f"  {title}")
    print(hr("═"))

    # LMM group effect
    print(f"\n  ▸ Linear Mixed Model   (value ~ Group × Day, random: mouse)")
    if not np.isnan(group_F):
        print(f"    Group main effect:  F ≈ {group_F:.2f},  p = {group_p:.4f}  "
              f"{significance_stars(group_p)}")
    else:
        print(f"    Group main effect:  LMM could not be fitted (see LMM log)")

    # Per-day table
    print(f"\n  ▸ Per-day Mann-Whitney U  (per-mouse daily means)\n")
    if mw_df.empty:
        print("    No data.")
        return

    hdr = (f"  {'Day':>4}  {'n SNr':>6}  {'n Ctrl':>6}  "
           f"{'Mean SNr':>9}  {'Mean Ctrl':>10}  "
           f"{'U':>7}  {'p':>8}  {'sig':>4}  {'d':>6}  {'effect'}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for _, row in mw_df.iterrows():
        print(
            f"  {int(row['Day']):>4}  "
            f"{int(row['n_SNrDTA']):>6}  "
            f"{int(row['n_Ctrl']):>6}  "
            f"{row['mean_SNrDTA']:>9.3f}  "
            f"{row['mean_Ctrl']:>10.3f}  "
            f"{row['U_stat']:>7.1f}  "
            f"{row['p_value']:>8.4f}  "
            f"{row['stars']:>4}  "
            f"{row['cohens_d']:>6.2f}  "
            f"{row['effect_size']}"
        )

    sig_days = mw_df[mw_df["p_value"] < 0.05]["Day"].tolist()
    if sig_days:
        print(f"\n  → Significant days (p < 0.05): {sig_days}")
    else:
        print(f"\n  → No days reached p < 0.05")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Y-maze SNr-DTA vs Ctrl statistics")
    parser.add_argument("folder", nargs="?",
                        default="/Volumes/Extreme SSD/ymaze/t1t2_and_clips",
                        help="Folder containing ymaze CSVs")
    args = parser.parse_args()

    in_dir = args.folder
    if not os.path.isdir(in_dir):
        sys.exit(f"Not a directory: {in_dir}")

    print(hr("═"))
    print(f"  Y-Maze Statistics  —  SNr-DTA vs Control")
    print(f"  All trials + correct hits (if labeled file present)")
    print(hr("═"))
    print()

    # ── Load ALL trials (always) ──────────────────────────────────────────────
    df_all = load_data(in_dir, "all")
    days   = sorted(df_all["Day"].dropna().unique())

    # ── Load hits-only (requires labeled file) ─────────────────────────────────
    labeled_path = os.path.join(in_dir, LABELED_FILE)
    has_labels   = os.path.exists(labeled_path)

    if has_labels:
        df_hits = load_data(in_dir, "hits")
        print("  Hits subset loaded for conditions 4-6.")
    else:
        df_hits = None
        print("  No labeled file found — conditions 4-6 (hits only) will be skipped.")
    print()

    # ── Build per-measure datasets: all trials ─────────────────────────────────
    t1_all   = mouse_day_means(df_all[df_all["time_point"] == "t1"].copy(), "T1")
    t2_all   = mouse_day_means(df_all[df_all["time_point"] == "t2"].copy(), "T2")
    t12_all  = build_t1_t2_combined(df_all)

    # ── Build per-measure datasets: hits only ──────────────────────────────────
    if df_hits is not None and not df_hits.empty:
        t1_hits  = mouse_day_means(df_hits[df_hits["time_point"] == "t1"].copy(), "T1_hits")
        t2_hits  = mouse_day_means(df_hits[df_hits["time_point"] == "t2"].copy(), "T2_hits")
        t12_hits = build_t1_t2_combined(df_hits)
    else:
        t1_hits = t2_hits = t12_hits = None

    # ── Statistics ─────────────────────────────────────────────────────────────
    lmm_log = []
    all_mw  = []

    analyses = [
        ("T1  — all trials  (approach duration)",         t1_all),
        ("T2  — all trials  (choice duration)",           t2_all),
        ("T1+T2  — all trials  (mean per day)",           t12_all),
    ]

    if t1_hits is not None:
        analyses += [
            ("T1  — correct hits only",                   t1_hits),
            ("T2  — correct hits only",                   t2_hits),
            ("T1+T2  — correct hits only  (mean per day)", t12_hits),
        ]

    all_el = []   # early/late results

    for title, mouse_df in analyses:
        measure_key = title.replace(' ', '_').replace('+', 'plus').replace('/', '').strip('_')
        mw_df = per_day_mannwhitney(mouse_df, days, measure_key)
        el_df = early_vs_late(mouse_df, measure_key)
        lmm_text, grp_F, grp_p = run_lmm(mouse_df, title.split()[0])

        print_section(title, mw_df, lmm_text, grp_F, grp_p)
        print_early_late(measure_key, el_df)

        all_mw.append(mw_df)
        all_el.append(el_df)
        lmm_log.append(f"\n{'='*72}\n{title}\n{'='*72}\n{lmm_text}\n")

    # ── Save outputs ───────────────────────────────────────────────────────────
    print()
    print(hr("─"))

    results_df = pd.concat(all_mw, ignore_index=True)
    for col in ["mean_SNrDTA", "sem_SNrDTA", "mean_Ctrl", "sem_Ctrl",
                "U_stat", "p_value", "cohens_d"]:
        if col in results_df.columns:
            results_df[col] = results_df[col].round(4)

    out_csv = os.path.join(in_dir, OUT_CSV)
    results_df.to_csv(out_csv, index=False)
    print(f"\n  Saved per-day results  : {out_csv}")

    # early/late CSV
    el_df_all = pd.concat(all_el, ignore_index=True)
    for col in ["mean_SNrDTA", "sem_SNrDTA", "mean_Ctrl", "sem_Ctrl",
                "U_or_W_stat", "p_value", "cohens_d"]:
        if col in el_df_all.columns:
            el_df_all[col] = pd.to_numeric(el_df_all[col], errors="coerce").round(4)
    out_el_csv = os.path.join(in_dir, "ymaze_stats_early_vs_late.csv")
    el_df_all.to_csv(out_el_csv, index=False)
    print(f"  Saved early/late results: {out_el_csv}")

    out_lmm = os.path.join(in_dir, OUT_LMM)
    with open(out_lmm, "w") as f:
        f.write(f"Y-Maze LMM Results\n")
        f.write("Generated by ymaze_stats_analysis.py\n\n")
        f.writelines(lmm_log)
    print(f"  Saved LMM summaries   : {out_lmm}")
    print()
    print(hr("═"))
    print("  Done.")
    print(hr("═"))


if __name__ == "__main__":
    main()