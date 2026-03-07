"""
dataset_statistics.py
Reproduces dataset statistics reported in Section 3 from the canonical
combined dataset (question-level and variant-level breakdowns).

Usage:
    python experiment_artifacts/analysis_scripts/dataset_statistics.py
    python experiment_artifacts/analysis_scripts/dataset_statistics.py --csv
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_PATH   = PROJECT_DIR / "experiment_artifacts" / "data" / "combined_dataset.csv"
STATS_OUT   = Path(__file__).parent / "dataset_stats.csv"

SOURCE_MAP = {
    "WVS":                   "WVS",
    "EVS":                   "EVS",
    "Pew_American_Trends":   "OpinionQA",
    "Understanding Society":  "Understanding Society",
    "steer-qa":              "OpinionQA",
}


def classify_source(raw: str) -> str:
    for prefix, label in SOURCE_MAP.items():
        if raw.startswith(prefix) or prefix in raw:
            return label
    return "Other"


def pct(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%"


# ============================================================================
# MAIN
# ============================================================================

def main(write_csv: bool = False) -> None:
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["n_options"] = df["scale_type"].str.extract(r"^(\d+)").astype(int)

    q_df = df.drop_duplicates(subset="question_id").copy()
    q_df["source_group"] = q_df["source"].map(classify_source)

    n_q = len(q_df)
    n_v = len(df)

    SEP  = "=" * 65
    SEP2 = "-" * 65

    print(SEP)
    print("  Dataset Statistics Report")
    print(SEP)

    # ---- Overview ----
    print(f"\n{SEP2}\n  Overview\n{SEP2}")
    print(f"  Unique questions : {n_q:>6,}   [thesis: 1,494]")
    print(f"  Total variants   : {n_v:>6,}   [thesis: 7,470]")
    print(f"  Variants / q     : {n_v / n_q:>6.1f}   [expected: 5.0]")

    # ---- Source distribution ----
    print(f"\n{SEP2}\n  Source distribution  [question level]\n{SEP2}")
    src_counts = q_df["source_group"].value_counts()
    print(f"  {'Source':<30} {'N':>6}  {'%':>7}")
    print(f"  {'-'*45}")
    for label, count in src_counts.items():
        print(f"  {label:<30} {count:>6,}  {pct(count, n_q):>7}")
    print(f"  {'TOTAL':<30} {src_counts.sum():>6,}")

    # ---- Subject categories ----
    print(f"\n{SEP2}\n  Subject categories  [question level]\n{SEP2}")
    subj_counts = q_df["subject"].value_counts()
    print(f"  Total unique categories: {len(subj_counts)}   [thesis: 34]")
    print(f"\n  {'Subject':<52} {'N':>5}  {'%':>7}")
    print(f"  {'-'*65}")
    for subj, count in subj_counts.items():
        print(f"  {subj:<52} {count:>5,}  {pct(count, n_q):>7}")

    # ---- Question types ----
    print(f"\n{SEP2}\n  Question types  [question level]\n{SEP2}")
    qt_counts = q_df["question_type"].value_counts()
    print(f"  Total unique types: {len(qt_counts)}   [thesis: 15]")
    print(f"\n  {'Type':<25} {'N':>6}  {'%':>7}")
    print(f"  {'-'*45}")
    for qt, count in qt_counts.items():
        print(f"  {qt:<25} {count:>6,}  {pct(count, n_q):>7}")

    # ---- Scale structure ----
    print(f"\n{SEP2}\n  Scale structure  [variant level, N = {n_v:,}]\n{SEP2}")
    opt_counts = df["n_options"].value_counts().sort_index()
    print(f"  {'Scale':<18} {'Variants':>8}  {'%':>7}")
    print(f"  {'-'*40}")
    for n_opt, count in opt_counts.items():
        print(f"  {n_opt}-option{'':<11} {count:>8,}  {pct(count, n_v):>7}")

    polarity_counts = df["scale_type"].value_counts().sort_index()
    print(f"\n  {'Scale type':<18} {'Variants':>8}  {'%':>7}")
    print(f"  {'-'*40}")
    for stype, count in polarity_counts.items():
        print(f"  {stype:<18} {count:>8,}  {pct(count, n_v):>7}")

    print(f"\n{SEP}\n  Report complete.\n{SEP}")

    # ---- Optional CSV export ----
    if write_csv:
        rows = []
        for label, count in src_counts.items():
            rows.append({"dimension": "source", "category": label,
                         "n_questions": count, "n_variants": count * 5,
                         "pct_questions": round(count / n_q * 100, 2)})
        for qt, count in qt_counts.items():
            rows.append({"dimension": "question_type", "category": qt,
                         "n_questions": count, "n_variants": count * 5,
                         "pct_questions": round(count / n_q * 100, 2)})
        for subj, count in subj_counts.items():
            rows.append({"dimension": "subject", "category": subj,
                         "n_questions": count, "n_variants": count * 5,
                         "pct_questions": round(count / n_q * 100, 2)})
        for stype, count in polarity_counts.items():
            rows.append({"dimension": "scale_type", "category": stype,
                         "n_questions": None, "n_variants": count,
                         "pct_questions": None})
        pd.DataFrame(rows).to_csv(STATS_OUT, index=False)
        print(f"\n  CSV written to: {STATS_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset statistics report.")
    parser.add_argument("--csv", action="store_true",
                        help="Also write statistics to dataset_stats.csv")
    main(write_csv=parser.parse_args().csv)
