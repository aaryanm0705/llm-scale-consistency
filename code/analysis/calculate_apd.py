"""
apd_calculation.py
Computes per-question APD for each model under Vanilla zero-shot and Expert
one-shot, then identifies persistently inconsistent, harder subset, and
robustly consistent question groups with representative examples.

Usage:
    python experiment_artifacts/analysis_scripts/apd_calculation.py
"""

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path(__file__).parent.parent / "results"

VANILLA_ZS = RESULTS_DIR / "consistency_results_vanilla_zeroshot.csv"
EXPERT_OS  = RESULTS_DIR / "consistency_results_expert_oneshot.csv"

MODEL_ORDER = [
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Qwen2.5-7B-Instruct",
    "gemma-2-9b-it",
    "deepseek-llm-7b-chat",
    "glm-4-9b-chat-hf",
]

SHORT = {
    "Llama-3.1-8B-Instruct":    "Llama-3.1",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen2.5-7B-Instruct":      "Qwen2.5",
    "gemma-2-9b-it":            "Gemma-2",
    "deepseek-llm-7b-chat":     "DeepSeek",
    "glm-4-9b-chat-hf":         "GLM-4",
}

# ============================================================================
# APD COMPUTATION
# ============================================================================

def compute_apd_per_question(df):
    records = []
    for (qid, model), grp in df.groupby(["question_id", "model"]):
        scores = grp.set_index("answer_var_id")["answer_score"].to_dict()
        var_ids = sorted(scores.keys())
        if len(var_ids) < 2:
            apd = 0.0
        else:
            diffs = [abs(scores[a] - scores[b]) for a, b in combinations(var_ids, 2)]
            apd = float(np.mean(diffs))
        records.append({"question_id": qid, "model": model, "apd": apd})
    return pd.DataFrame(records)


def load_question_meta(df):
    return (df.drop_duplicates("question_id")
              .set_index("question_id")[["question", "subject", "question_type", "source"]])


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading CSVs...")
    vzs = pd.read_csv(VANILLA_ZS)
    eos = pd.read_csv(EXPERT_OS)

    meta = load_question_meta(vzs)

    print("Computing per-question APD...")
    apd_vzs = compute_apd_per_question(vzs)
    apd_eos = compute_apd_per_question(eos)

    total_questions = apd_vzs["question_id"].nunique()

    # ---- Per-model perfect-consistency rate (Vanilla zero-shot) ----
    print(f"\n=== Perfect-consistency rates under Vanilla zero-shot (APD = 0) ===")
    print(f"Total questions: {total_questions}")
    for model in MODEL_ORDER:
        sub = apd_vzs[apd_vzs["model"] == model]
        n_perfect = (sub["apd"] == 0.0).sum()
        print(f"  {SHORT[model]}: {n_perfect} / {len(sub)} ({100 * n_perfect / len(sub):.1f}%)")

    wide_vzs = apd_vzs.pivot(index="question_id", columns="model", values="apd")
    wide_eos = apd_eos.pivot(index="question_id", columns="model", values="apd")

    # ---- Persistently inconsistent: all 6 models APD > 0 under Vanilla zero-shot ----
    persist = wide_vzs[(wide_vzs[MODEL_ORDER] > 0).all(axis=1)].copy()
    n_persist = len(persist)
    print(f"\n=== Persistently inconsistent (all 6 models APD > 0, Vanilla zero-shot) ===")
    print(f"  Count: {n_persist} / {total_questions} ({100 * n_persist / total_questions:.1f}%)")

    persist["mean_apd"] = persist[MODEL_ORDER].mean(axis=1)
    print("  Top 5 examples (highest mean APD across all 6 models):")
    for qid, row in persist.nlargest(5, "mean_apd").iterrows():
        q = meta.loc[qid]
        print(f"    Q{qid}: [{q['subject']} | {q['question_type']}]")
        print(f"      Text: {q['question'][:100]}")
        for m in MODEL_ORDER:
            print(f"      {SHORT[m]}: APD={row[m]:.3f}", end="  ")
        print(f"\n      Mean APD={row['mean_apd']:.3f}")

    # ---- Harder subset: also APD > 0 for all 6 models under Expert one-shot ----
    common = persist.index.intersection(wide_eos.index)
    harder = wide_eos.loc[common, MODEL_ORDER]
    harder = harder[(harder > 0).all(axis=1)].copy()
    n_harder = len(harder)
    print(f"\n=== Harder subset (also APD > 0, all 6 models, Expert one-shot) ===")
    print(f"  Count: {n_harder} / {total_questions} ({100 * n_harder / total_questions:.1f}%)")

    harder["mean_apd_eos"] = harder[MODEL_ORDER].mean(axis=1)
    print("  Top 3 examples (highest mean APD under Expert one-shot):")
    for qid, row in harder.nlargest(3, "mean_apd_eos").iterrows():
        q = meta.loc[qid]
        vzs_row = persist.loc[qid]
        print(f"    Q{qid}: [{q['subject']} | {q['question_type']}]")
        print(f"      Text: {q['question'][:100]}")
        print(f"      VZS APD: ", end="")
        for m in MODEL_ORDER:
            print(f"{SHORT[m]}={vzs_row[m]:.3f}", end="  ")
        print(f"\n      EOS APD: ", end="")
        for m in MODEL_ORDER:
            print(f"{SHORT[m]}={row[m]:.3f}", end="  ")
        print()

    # ---- Robustly consistent: all 6 models APD = 0 under Vanilla zero-shot ----
    robust = wide_vzs[(wide_vzs[MODEL_ORDER] == 0.0).all(axis=1)]
    n_robust = len(robust)
    print(f"\n=== Robustly consistent (all 6 models APD = 0, Vanilla zero-shot) ===")
    print(f"  Count: {n_robust} / {total_questions} ({100 * n_robust / total_questions:.1f}%)")

    print("  Sample examples:")
    for qid, _ in robust.sample(min(5, n_robust), random_state=42).iterrows():
        q = meta.loc[qid]
        print(f"    Q{qid}: [{q['subject']} | {q['question_type']}]")
        print(f"      Text: {q['question'][:100]}")

    # ---- Summary ----
    print(f"\n=== SUMMARY ===")
    print(f"Total questions                                   : {total_questions}")
    print(f"Persistently inconsistent (all models, VZS)      : {n_persist} ({100 * n_persist / total_questions:.1f}%)")
    print(f"Harder subset (also inconsistent, Expert one-shot): {n_harder} ({100 * n_harder / total_questions:.1f}%)")
    print(f"Robustly consistent (all models, VZS)            : {n_robust} ({100 * n_robust / total_questions:.1f}%)")


if __name__ == "__main__":
    main()
