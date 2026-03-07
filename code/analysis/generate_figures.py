"""
generate_figures.py
Generates all thesis figures for Chapters 5 and 6.

Outputs (saved to figures/):
    fig_5_apd_schematic_v2.png
    fig_6_1_apd_heatmap.pdf
    fig_6_2_apd_distribution_vanilla_zeroshot.png
    fig_6_3_strategy_comparison_zeroshot.png
    fig_6_4_mode_effect.png
    fig_6_5_delta_heatmap.png

Usage:
    python experiment_artifacts/analysis_scripts/generate_figures.py
"""

import copy
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import seaborn as sns

# ============================================================================
# PATHS
# ============================================================================

PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_DIR / "experiment_artifacts" / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ============================================================================
# SHARED CONSTANTS
# ============================================================================

MODEL_ORDER = [
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Qwen2.5-7B-Instruct",
    "gemma-2-9b-it",
    "deepseek-llm-7b-chat",
    "glm-4-9b-chat-hf",
]

MODEL_DISPLAY = {
    "Llama-3.1-8B-Instruct":    "Llama-3.1",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen2.5-7B-Instruct":      "Qwen2.5",
    "gemma-2-9b-it":            "Gemma-2",
    "deepseek-llm-7b-chat":     "DeepSeek",
    "glm-4-9b-chat-hf":         "GLM-4",
}

# Order by ascending Vanilla Zero-Shot APD (used in heatmaps)
MODEL_ORDER_APD = ["DeepSeek", "Gemma-2", "Mistral-7B", "Llama-3.1", "Qwen2.5", "GLM-4"]

# Alphabetical sort for legend / bar chart consistency
DISPLAY_LABELS_SORTED = ["DeepSeek", "Gemma-2", "GLM-4", "Llama-3.1", "Mistral-7B", "Qwen2.5"]
MODEL_COLOR = dict(zip(DISPLAY_LABELS_SORTED, sns.color_palette("colorblind", 6)))

RESULT_FILES = {
    "vanilla-zeroshot":        RESULTS_DIR / "consistency_results_vanilla_zeroshot.csv",
    "vanilla-oneshot":         RESULTS_DIR / "consistency_results_vanilla_oneshot.csv",
    "vanilla-sequential":      RESULTS_DIR / "consistency_results_vanilla_sequential.csv",
    "expert-zeroshot":         RESULTS_DIR / "consistency_results_expert_zeroshot.csv",
    "expert-oneshot":          RESULTS_DIR / "consistency_results_expert_oneshot.csv",
    "expert-sequential":       RESULTS_DIR / "consistency_results_expert_sequential.csv",
    "inconsistent-zeroshot":   RESULTS_DIR / "consistency_results_inconsistent_zeroshot.csv",
    "inconsistent-oneshot":    RESULTS_DIR / "consistency_results_inconsistent_oneshot.csv",
    "inconsistent-sequential": RESULTS_DIR / "consistency_results_inconsistent_sequential.csv",
    "scale-anchored":          RESULTS_DIR / "consistency_results_scale_anchored.csv",
    "cot-zeroshot":            RESULTS_DIR / "consistency_results_cot_zeroshot.csv",
    "cot-oneshot":             RESULTS_DIR / "consistency_results_cot_oneshot.csv",
    "cot-sequential":          RESULTS_DIR / "consistency_results_cot_sequential.csv",
}

# ============================================================================
# GLOBAL STYLE
# ============================================================================

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family":           "serif",
    "font.size":             10,
    "axes.titlesize":        13,
    "axes.labelsize":        11,
    "xtick.labelsize":       10,
    "ytick.labelsize":       10,
    "legend.fontsize":       10,
    "legend.title_fontsize": 10,
    "figure.dpi":            150,
})

W_TEXT = 6.27  # full text width (inches)

# ============================================================================
# DATA LOADING AND APD COMPUTATION
# ============================================================================

def compute_apd_per_question(df):
    records = []
    for (qid, mdl), grp in df.groupby(["question_id", "model"]):
        scores = [s for s in grp["answer_score"].values if pd.notna(s)]
        if len(scores) < 2:
            continue
        apd = np.mean([abs(a - b) for a, b in itertools.combinations(scores, 2)])
        records.append({"question_id": qid, "model": mdl, "apd": apd})
    return pd.DataFrame(records)


print("Loading results...")
dfs = {}
for key, path in RESULT_FILES.items():
    if path.exists():
        df = pd.read_csv(path)
        df["model_display"] = df["model"].map(MODEL_DISPLAY)
        dfs[key] = df

missing = [k for k in RESULT_FILES if k not in dfs]
if missing:
    print(f"  Missing configs: {missing}")

print("Computing per-question APD...")
apd_by_config = {}
for key, df in dfs.items():
    apd_df = compute_apd_per_question(df)
    apd_df["model_display"] = apd_df["model"].map(MODEL_DISPLAY)
    apd_by_config[key] = apd_df

mean_apd_by_cfg = {
    key: df.groupby("model_display")["apd"].mean()
    for key, df in apd_by_config.items()
}
baseline = mean_apd_by_cfg.get("vanilla-zeroshot", pd.Series(dtype=float))


# ============================================================================
# FIGURE 5 — APD Computation Schematic
# ============================================================================

print("Generating Figure 5 (APD schematic)...")

plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150})

SCORE   = [-1.000, -1.0 / 3.0, +1.0 / 3.0, +1.000]
LETTERS = ["A", "B", "C", "D"]
VARIANTS = ["V1", "V2", "V3", "V4", "V5"]
LABEL_SETS = [
    ["Fully agree",      "Somewhat agree",    "Somewhat disagree",  "Fully disagree"],
    ["Strongly agree",   "Agree",             "Disagree",           "Strongly disagree"],
    ["Absolutely agree", "Partly agree",      "Partly disagree",    "Absolutely disagree"],
    ["Completely agree", "Somewhat agree",    "Somewhat disagree",  "Completely disagree"],
    ["Totally agree",    "More agree",        "More disagree",      "Totally disagree"],
]
QUESTION_TEXT = '"Should environmental protection take priority over economic growth?"'
SEL_A = [2, 2, 2, 2, 2]
SEL_B = [1, 2, 1, 2, 3]

C_BLUE      = "#2171b5"
C_RED       = "#d62728"
C_PURPLE    = "#7b2d8b"
C_APD0      = "#27ae60"
C_APDN      = "#c0392b"
C_ROW       = "#f4f4f4"
C_HDR_BG    = "#e8eaed"
C_PANEL_HDR = "#2c3e50"


def _add_rounded_rect(ax, x, y, w, h, fc, ec="none", lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008",
        fc=fc, ec=ec, linewidth=lw,
    ))


def _add_plain_rect(ax, x, y, w, h, fc):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc=fc, ec="none", zorder=0))


def _draw_schematic_panel(ax, sel_indices, is_consistent):
    title = "(a)  Perfect Consistency" if is_consistent else "(b)  Scale-Sensitive Response"
    ax.text(0.5, 0.965, title, fontsize=10.5, fontweight="bold",
            ha="center", va="top", color=C_PANEL_HDR)
    ax.text(0.5, 0.895, QUESTION_TEXT, fontsize=8.5, ha="center",
            va="top", style="italic", color="#444444")

    hy = 0.805
    _add_plain_rect(ax, 0.01, hy - 0.030, 0.98, 0.050, C_HDR_BG)
    ax.text(0.07,  hy, "Variant",         fontsize=8.5, ha="center", va="center",
            fontweight="bold", color="#444444")
    ax.text(0.435, hy, "Selected answer", fontsize=8.5, ha="center", va="center",
            fontweight="bold", color="#444444")
    ax.text(0.915, hy, "Score",           fontsize=8.5, ha="center", va="center",
            fontweight="bold", color="#444444")
    ax.axhline(hy - 0.030, xmin=0.01, xmax=0.99, color="#b0b0b0", lw=0.8)

    ROW_H = 0.072
    ROW_GAP = 0.005
    first_row_y = hy - 0.062
    row_y_centers = [first_row_y - i * (ROW_H + ROW_GAP) for i in range(5)]

    scores = []
    for i, (vname, labels, sel_idx) in enumerate(zip(VARIANTS, LABEL_SETS, sel_indices)):
        yc = row_y_centers[i]
        _add_plain_rect(ax, 0.01, yc - ROW_H / 2, 0.98, ROW_H,
                        fc=C_ROW if i % 2 == 0 else "white")
        score = SCORE[sel_idx]
        scores.append(score)
        badge_c = C_BLUE if is_consistent else {1: C_RED, 2: C_BLUE, 3: C_PURPLE}.get(sel_idx, C_BLUE)
        _add_rounded_rect(ax, 0.135, yc - 0.023, 0.046, 0.046, fc=badge_c)
        ax.text(0.07, yc, vname, fontsize=8.5, ha="center", va="center",
                color="#333333", fontweight="bold")
        ax.text(0.158, yc, LETTERS[sel_idx], fontsize=8.5, ha="center", va="center",
                color="white", fontweight="bold")
        ax.text(0.21, yc, labels[sel_idx], fontsize=8.5, ha="left", va="center", color="#333333")
        ax.text(0.915, yc, f"{score:+.3f}", fontsize=8.5, ha="center", va="center",
                color="#333333", fontfamily="monospace")

    last_bottom = row_y_centers[-1] - ROW_H / 2
    div_y = last_bottom - 0.018
    ax.axhline(div_y, xmin=0.01, xmax=0.99, color="#cccccc", lw=0.8)

    comp_y = div_y - 0.035
    if is_consistent:
        ax.text(0.5, comp_y,
                r"All 10 pairwise distances: $|s_i - s_j| = |{+0.333} - {+0.333}| = 0.000$ for all $(i,j)$",
                fontsize=8.5, ha="center", va="top", color="#555555")
    else:
        ax.text(0.5, comp_y, "Exemplar pairwise distances:",
                fontsize=8.5, ha="center", va="top", color="#555555")
        ax.text(0.5, comp_y - 0.055,
                r"$|s_1 - s_2| = |-0.333 - (+0.333)| = 0.667"
                r"\quad |s_1 - s_5| = |-0.333 - (+1.000)| = 1.333\quad \ldots$",
                fontsize=8.5, ha="center", va="top", color="#555555")

    pairs = list(itertools.combinations(scores, 2))
    apd = np.mean([abs(a - b) for a, b in pairs])
    box_fc = "#e8f5e9" if is_consistent else "#fdecea"
    box_ec = C_APD0 if is_consistent else C_APDN
    _add_rounded_rect(ax, 0.15, 0.06, 0.70, 0.115, fc=box_fc, ec=box_ec, lw=1.8)
    ax.text(0.5, 0.118,
            rf"$\mathrm{{APD}} = \frac{{1}}{{10}}\sum_{{(i,j)}} |s_i - s_j| = {apd:.3f}$",
            fontsize=9.5, ha="center", va="center", color=box_ec, fontweight="bold")


fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(W_TEXT, 6.80), gridspec_kw={"hspace": 0.10})
for ax in (ax5a, ax5b):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

_draw_schematic_panel(ax5a, SEL_A, is_consistent=True)
_draw_schematic_panel(ax5b, SEL_B, is_consistent=False)
fig5.tight_layout(pad=0.4)
out5 = FIGURES_DIR / "fig_5_apd_schematic_v2.png"
fig5.savefig(out5, dpi=300, bbox_inches="tight")
plt.close(fig5)
print(f"  Saved: {out5}")

# Restore global style after schematic's custom rcParams
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
})


# ============================================================================
# FIGURE 6.1 — Mean APD Heatmap (10 non-CoT configs x 6 models)
# ============================================================================

print("Generating Figure 6.1 (APD heatmap)...")

HEATMAP_CONFIGS = {
    "Vanilla\nzero-shot":        "vanilla-zeroshot",
    "Vanilla\none-shot":         "vanilla-oneshot",
    "Vanilla\nchainwise":        "vanilla-sequential",
    "Expert\nzero-shot":         "expert-zeroshot",
    "Expert\none-shot":          "expert-oneshot",
    "Expert\nchainwise":         "expert-sequential",
    "Inconsistent\nzero-shot":   "inconsistent-zeroshot",
    "Inconsistent\none-shot":    "inconsistent-oneshot",
    "Inconsistent\nchainwise":   "inconsistent-sequential",
    "Scale-\nAnchored":          "scale-anchored",
}

config_labels = list(HEATMAP_CONFIGS.keys())
model_labels  = [MODEL_DISPLAY[m] for m in MODEL_ORDER]

data61 = np.full((len(config_labels), len(MODEL_ORDER)), np.nan)
for i, (cfg_label, cfg_key) in enumerate(HEATMAP_CONFIGS.items()):
    if cfg_key in mean_apd_by_cfg:
        for j, model in enumerate(MODEL_ORDER):
            short = MODEL_DISPLAY[model]
            val = mean_apd_by_cfg[cfg_key].get(short, np.nan)
            data61[i, j] = val

fig61, ax61 = plt.subplots(figsize=(W_TEXT, 4.5))
im = ax61.imshow(data61, cmap=plt.cm.YlOrRd, vmin=0.0, vmax=0.80, aspect="auto")

for i in range(len(config_labels)):
    for j in range(len(model_labels)):
        val = data61[i, j]
        if not np.isnan(val):
            color = "white" if (val / 0.80) > 0.65 else "black"
            ax61.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

ax61.set_xticks(range(len(model_labels)))
ax61.set_xticklabels(model_labels, fontsize=8)
ax61.set_yticks(range(len(config_labels)))
ax61.set_yticklabels(config_labels, fontsize=8, linespacing=0.9)
ax61.tick_params(axis="both", which="both", length=0)

for y in [2.5, 5.5, 8.5]:
    ax61.axhline(y, color="white", linewidth=1.8)

cbar = fig61.colorbar(im, ax=ax61, fraction=0.03, pad=0.02)
cbar.set_label("Mean APD", fontsize=8)
cbar.ax.tick_params(labelsize=7)
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax61.set_title("Mean APD by model and evaluation configuration", fontsize=9, pad=6)

fig61.tight_layout()
out61 = FIGURES_DIR / "fig_6_1_apd_heatmap.pdf"
fig61.savefig(out61, format="pdf", bbox_inches="tight", dpi=300)
plt.close(fig61)
print(f"  Saved: {out61}")


# ============================================================================
# FIGURE 6.2 — APD Distribution by Model (Violin + Strip, Vanilla Zero-Shot)
# ============================================================================

print("Generating Figure 6.2 (violin plot)...")

apd_vanilla = apd_by_config["vanilla-zeroshot"].copy()

model_order_fig62 = (
    apd_vanilla.groupby("model_display")["apd"]
    .mean().sort_values().index.tolist()
)
pct_perfect = (
    apd_vanilla.groupby("model_display")
    .apply(lambda g: (g["apd"] == 0).sum() / len(g) * 100)
    .reindex(model_order_fig62)
)

fig62, ax62 = plt.subplots(figsize=(7, 4.5))

vp = ax62.violinplot(
    [apd_vanilla.loc[apd_vanilla["model_display"] == m, "apd"].values
     for m in model_order_fig62],
    positions=range(len(model_order_fig62)),
    showmedians=False, showextrema=False,
)
for body, m in zip(vp["bodies"], model_order_fig62):
    body.set_facecolor(MODEL_COLOR[m])
    body.set_edgecolor("black")
    body.set_linewidth(0.6)
    body.set_alpha(0.7)

rng = np.random.default_rng(42)
for i, m in enumerate(model_order_fig62):
    vals = apd_vanilla.loc[apd_vanilla["model_display"] == m, "apd"].values
    jitter = rng.uniform(-0.18, 0.18, size=len(vals))
    ax62.scatter(i + jitter, vals, color=MODEL_COLOR[m], alpha=0.15, s=1.5, zorder=2)
    ax62.scatter(i, np.median(vals), color="white", s=20, zorder=5,
                 edgecolors="black", linewidths=0.6)
    ax62.text(i, 1.92, f"{pct_perfect[m]:.1f}%\nperfect",
              ha="center", va="bottom", fontsize=9, color="black")

ax62.set_xticks(range(len(model_order_fig62)))
ax62.set_xticklabels(model_order_fig62)
ax62.set_ylabel("APD")
ax62.set_ylim(0, 2.15)
ax62.set_xlim(-0.6, len(model_order_fig62) - 0.4)

fig62.tight_layout()
out62 = FIGURES_DIR / "fig_6_2_apd_distribution_vanilla_zeroshot.png"
fig62.savefig(out62, dpi=300, bbox_inches="tight")
plt.close(fig62)
print(f"  Saved: {out62}")


# ============================================================================
# FIGURE 6.3 — Mean APD by Strategy (Zero-Shot Configs, Grouped Bar)
# ============================================================================

print("Generating Figure 6.3 (strategy comparison bar chart)...")

ZEROSHOT_CONFIGS = [
    ("vanilla-zeroshot",      "Vanilla"),
    ("expert-zeroshot",       "Expert"),
    ("scale-anchored",        "Scale-Anchored"),
    ("inconsistent-zeroshot", "Inconsistent"),
]

rows63 = []
for cfg_key, cfg_label in ZEROSHOT_CONFIGS:
    if cfg_key not in apd_by_config:
        continue
    agg = (apd_by_config[cfg_key]
           .groupby("model_display")["apd"]
           .agg(["mean", "std"])
           .reset_index()
           .rename(columns={"mean": "mean_apd", "std": "std_apd"}))
    agg["strategy"] = cfg_label
    rows63.append(agg)

df63 = pd.concat(rows63, ignore_index=True)
strategy_order = ["Vanilla", "Expert", "Scale-Anchored", "Inconsistent"]
bar_width  = 0.12
group_gap  = 0.1
group_width = len(DISPLAY_LABELS_SORTED) * bar_width + group_gap
x_centers  = np.arange(len(strategy_order)) * group_width
offsets    = np.linspace(
    -(len(DISPLAY_LABELS_SORTED) - 1) / 2 * bar_width,
     (len(DISPLAY_LABELS_SORTED) - 1) / 2 * bar_width,
    len(DISPLAY_LABELS_SORTED)
)

fig63, ax63 = plt.subplots(figsize=(7, 4.2))

for j, mdl in enumerate(DISPLAY_LABELS_SORTED):
    means, stds = [], []
    for strat in strategy_order:
        row = df63.loc[(df63["strategy"] == strat) & (df63["model_display"] == mdl)]
        means.append(row["mean_apd"].values[0] if len(row) else np.nan)
        stds.append(row["std_apd"].values[0] if len(row) else np.nan)
    ax63.bar(x_centers + offsets[j], means, width=bar_width,
             color=MODEL_COLOR[mdl], label=mdl,
             yerr=stds, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "black"},
             zorder=3)

vanilla_grand_mean = df63.loc[df63["strategy"] == "Vanilla", "mean_apd"].mean()
ax63.axhline(vanilla_grand_mean, color="black", linestyle="--", linewidth=0.9, zorder=4)
ax63.text(x_centers[-1] + group_width * 0.45, vanilla_grand_mean + 0.005,
          "Vanilla baseline", fontsize=9, va="bottom", ha="right")

ax63.set_xticks(x_centers)
ax63.set_xticklabels(strategy_order)
ax63.set_ylabel("Mean APD")
ax63.set_ylim(0, None)
ax63.legend(title="Model", loc="upper right", fontsize=9,
            title_fontsize=9, framealpha=0.9, ncol=1)

fig63.tight_layout()
out63 = FIGURES_DIR / "fig_6_3_strategy_comparison_zeroshot.png"
fig63.savefig(out63, dpi=300, bbox_inches="tight")
plt.close(fig63)
print(f"  Saved: {out63}")


# ============================================================================
# FIGURE 6.4 — Mode Effect: Zero-Shot -> One-Shot -> Sequential
# ============================================================================

print("Generating Figure 6.4 (mode effect line plot)...")

MODE_CONFIGS = {
    "Vanilla": [
        ("vanilla-zeroshot",   "Zero-Shot"),
        ("vanilla-oneshot",    "One-Shot"),
        ("vanilla-sequential", "Chainwise"),
    ],
    "Expert": [
        ("expert-zeroshot",   "Zero-Shot"),
        ("expert-oneshot",    "One-Shot"),
        ("expert-sequential", "Chainwise"),
    ],
    "Inconsistent": [
        ("inconsistent-zeroshot",   "Zero-Shot"),
        ("inconsistent-oneshot",    "One-Shot"),
        ("inconsistent-sequential", "Chainwise"),
    ],
    "CoT": [
        ("cot-zeroshot",   "Zero-Shot"),
        ("cot-oneshot",    "One-Shot"),
        ("cot-sequential", "Chainwise"),
    ],
}

all_vals = [
    mean_apd_by_cfg[cfg_key].values.tolist()
    for configs in MODE_CONFIGS.values()
    for cfg_key, _ in configs
    if cfg_key in mean_apd_by_cfg
]
y_max = max(v for vals in all_vals for v in vals) * 1.08

fig64, axes64 = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
axes64_flat = axes64.flatten()

for ax, (strat, configs) in zip(axes64_flat, MODE_CONFIGS.items()):
    for mdl in DISPLAY_LABELS_SORTED:
        y_vals = []
        for cfg_key, _ in configs:
            if cfg_key not in mean_apd_by_cfg:
                y_vals.append(np.nan)
                continue
            y_vals.append(mean_apd_by_cfg[cfg_key].get(mdl, np.nan))
        ax.plot(range(3), y_vals, color=MODEL_COLOR[mdl],
                marker="o", markersize=6, linewidth=1.5, label=mdl, zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Zero-Shot", "One-Shot", "Chainwise"])
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Mean APD" if ax in axes64[:, 0] else "")
    ax.text(0.04, 0.95, strat, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

handles = [mpatches.Patch(color=MODEL_COLOR[m], label=m) for m in DISPLAY_LABELS_SORTED]
fig64.legend(handles=handles, title="Model", loc="center right",
             fontsize=10, title_fontsize=10, framealpha=0.9, bbox_to_anchor=(1.13, 0.5))

fig64.tight_layout()
out64 = FIGURES_DIR / "fig_6_4_mode_effect.png"
fig64.savefig(out64, dpi=300, bbox_inches="tight")
plt.close(fig64)
print(f"  Saved: {out64}")


# ============================================================================
# FIGURE 6.5 — Delta-from-Baseline Heatmap (ΔAPD vs. Vanilla Zero-Shot)
# ============================================================================

print("Generating Figure 6.5 (delta heatmap)...")

DELTA_CONFIGS = [
    ("scale-anchored",          "Scale-Anchored"),
    ("vanilla-oneshot",         "Vanilla One-Shot"),
    ("vanilla-sequential",      "Vanilla Chainwise"),
    ("expert-zeroshot",         "Expert Zero-Shot"),
    ("expert-oneshot",          "Expert One-Shot"),
    ("expert-sequential",       "Expert Chainwise"),
    ("inconsistent-zeroshot",   "Inconsistent Zero-Shot"),
    ("inconsistent-oneshot",    "Inconsistent One-Shot"),
    ("inconsistent-sequential", "Inconsistent Chainwise"),
    ("cot-zeroshot",            "CoT Zero-Shot"),
    ("cot-oneshot",             "CoT One-Shot"),
    ("cot-sequential",          "CoT Chainwise"),
]

delta_rows = []
for cfg_key, cfg_label in DELTA_CONFIGS:
    series = mean_apd_by_cfg.get(cfg_key, pd.Series(dtype=float))
    row = {"Configuration": cfg_label}
    for mdl in MODEL_ORDER_APD:
        raw  = series.get(mdl, np.nan)
        base = baseline.get(mdl, np.nan)
        row[mdl] = (raw - base) if not (np.isnan(raw) or np.isnan(base)) else np.nan
    delta_rows.append(row)

delta_df = (pd.DataFrame(delta_rows)
            .set_index("Configuration")
            .reindex([c for _, c in DELTA_CONFIGS]))

annot_delta = delta_df[MODEL_ORDER_APD].applymap(
    lambda x: "" if pd.isna(x) else (f"+{x:.3f}" if x > 0 else f"{x:.3f}")
)

cmap_delta = copy.copy(plt.cm.RdBu_r)
cmap_delta.set_bad(color="#D3D3D3")

fig65, ax65 = plt.subplots(figsize=(10, 8))
sns.heatmap(delta_df[MODEL_ORDER_APD], annot=annot_delta, fmt="",
            cmap=cmap_delta, center=0, vmin=-0.50, vmax=+0.50,
            linewidths=0.5, linecolor="white", annot_kws={"size": 9},
            cbar_kws={"label": "ΔAPD (relative to Vanilla Zero-Shot)", "shrink": 0.8},
            ax=ax65)

ax65.set_xlabel("Model")
ax65.set_ylabel("Configuration")
ax65.set_xticklabels(MODEL_ORDER_APD, rotation=0)
ax65.set_yticklabels(ax65.get_yticklabels(), rotation=0)
ax65.hlines([1, 3, 6, 9], xmin=0, xmax=len(MODEL_ORDER_APD), colors="black", linewidths=2)

fig65.tight_layout()
out65 = FIGURES_DIR / "fig_6_5_delta_heatmap.png"
fig65.savefig(out65, dpi=300, bbox_inches="tight")
plt.close(fig65)
print(f"  Saved: {out65}")


# ============================================================================
# SUMMARY
# ============================================================================

print("\nDone.")
print(f"  {out5}")
print(f"  {out61}")
print(f"  {out62}")
print(f"  {out63}")
print(f"  {out64}")
print(f"  {out65}")
if missing:
    print(f"\n  Warning: missing configs: {missing}")
