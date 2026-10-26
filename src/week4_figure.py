"""Render the single Week 4 comparison figure from the actual aggregate metrics.

Reads ``data/week4_real_vs_synthetic.csv`` (produced by
``src/week4_real_vs_synthetic.py``) and draws ONE grouped bar chart comparing
SYNTHETIC vs REAL HumAID on the three label-dependent metrics:

    * exact / manual match rate
    * location extraction rate
    * missing-location rate

Only the two rows that have ground-truth labels are used (the SYNTHETIC set and
the labelled REAL_DATASET/HumAID subset); the unlabelled-corpus row is excluded
because its match/missing rates are ``NA``. No values are hardcoded or invented
-- everything is read straight from the CSV.

Outputs:
    results/week4_real_vs_synthetic.png
    results/week4_real_vs_synthetic_caption.txt
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "week4_real_vs_synthetic.csv"
RESULTS_DIR = ROOT / "results"
FIGURE_PATH = RESULTS_DIR / "week4_real_vs_synthetic.png"
CAPTION_PATH = RESULTS_DIR / "week4_real_vs_synthetic_caption.txt"

# Metric column -> short axis label.
METRICS = [
    ("exact_or_manual_location_match_rate", "Exact / manual\nmatch rate"),
    ("location_extraction_rate", "Location\nextraction rate"),
    ("missing_location_rate", "Missing-location\nrate"),
]


def _load_labelled_rows():
    """Return (synthetic_row, real_row) from the aggregate CSV.

    Picks the SYNTHETIC row and the labelled REAL_DATASET row (the one whose
    label basis is manual annotations, i.e. not the unlabelled corpus sample).
    """
    df = pd.read_csv(CSV_PATH)
    synthetic = df[df["data_origin"] == "SYNTHETIC"].iloc[0]
    real = df[
        (df["data_origin"] == "REAL_DATASET")
        & (df["label_basis"] == "manual_annotations")
    ].iloc[0]
    return synthetic, real


def build_figure():
    synthetic, real = _load_labelled_rows()

    labels = [label for _, label in METRICS]
    syn_values = [float(synthetic[col]) for col, _ in METRICS]
    real_values = [float(real[col]) for col, _ in METRICS]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars_syn = ax.bar(
        x - width / 2, syn_values, width,
        label=f"SYNTHETIC (n={int(synthetic['number_of_posts'])})",
        color="#4C72B0",
    )
    bars_real = ax.bar(
        x + width / 2, real_values, width,
        label=f"REAL — HumAID (n={int(real['number_of_posts'])})",
        color="#C44E52",
    )

    ax.set_ylabel("Percentage of posts (%)")
    ax.set_title(
        "Geographic location extraction: synthetic claims vs real HumAID crisis posts\n"
        "(same spaCy extractor; from data/week4_real_vs_synthetic.csv)",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bars in (bars_syn, bars_real):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)
    return synthetic, real, syn_values, real_values


def write_caption(synthetic, real, syn_values, real_values):
    syn_exact, syn_extract, syn_missing = syn_values
    real_exact, real_extract, real_missing = real_values
    caption = (
        "Figure: Geographic location extraction on synthetic claims vs real HumAID "
        "crisis posts.\n\n"
        f"The same spaCy geographic-entity extractor was run over {int(synthetic['number_of_posts'])} "
        f"manually written synthetic claims and {int(real['number_of_posts'])} real, manually "
        "labelled HumAID disaster tweets. Real crisis-language performance is substantially "
        "lower than synthetic performance across every metric: exact/manual match rate falls "
        f"from {syn_exact:.0f}% to {real_exact:.0f}%, extraction rate from {syn_extract:.0f}% to "
        f"{real_extract:.0f}%, while the missing-location rate rises from {syn_missing:.0f}% to "
        f"{real_missing:.0f}%. Because the extractor is identical, the gap is attributable to the "
        "text itself: real posts hide places in hashtags, abbreviations, nicknames, misspellings, "
        "and multi-location or consequence-only language. This demonstrates the importance of "
        "evaluating geographic extraction on noisy real-world text rather than on controlled "
        "synthetic claims, which otherwise overstate how well the pipeline's location step works. "
        "Values are read directly from data/week4_real_vs_synthetic.csv; no values are fabricated."
    )
    CAPTION_PATH.write_text(caption + "\n", encoding="utf-8")
    return caption


if __name__ == "__main__":
    synthetic, real, syn_values, real_values = build_figure()
    write_caption(synthetic, real, syn_values, real_values)
    print(f"Wrote {FIGURE_PATH}")
    print(f"Wrote {CAPTION_PATH}")
