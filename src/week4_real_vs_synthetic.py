"""Week 4 experiment: location extraction on REAL vs SYNTHETIC social-media text.

Research question
-----------------
*Does geographic entity extraction perform differently on realistic, noisy
social-media language (real disaster posts) compared with controlled synthetic
claims?*

Design
------
The same spaCy geographic-entity extractor (the preserved Week 1/2 baseline) is
run over two clearly-separated data categories and scored identically:

    A. SYNTHETIC    -- data/week3_location_test_synthetic.csv   (manual labels)
    B. REAL_DATASET -- HumAID real tweets (hydrated locally; manual labels)

An optional third row reports **extraction-only** statistics on the broader
HumAID corpus (``data/processed/humaid_claims.csv`` from ``humaid_loader.py``),
which has **no** ground-truth location labels -- so, honouring the research rule,
only extraction statistics are reported there and the match rate is ``NA``.

Metrics (per data category)
---------------------------
    number_of_posts
    location_extraction_rate                 -- share of posts a location is extracted from
    exact_or_manual_location_match_rate      -- share matching the manual label (where labels exist)
    ambiguous_location_rate                  -- share with multiple competing locations, no clear primary
    missing_location_rate                    -- share where a location is expected but none is extracted

Integrity notes
---------------
* No location labels are fabricated. The real-corpus row has no ground truth, so
  its match rate is ``NA`` rather than a made-up number.
* "Ambiguous" here is measured at the *extraction* level (several candidate
  places with no disambiguating cue), which needs no network. This is distinct
  from the geocoder-level ambiguity reported in the Week 3 evaluation.
* Extraction uses spaCy only, so this experiment is deterministic and offline.

Output
------
    data/week4_real_vs_synthetic.csv
"""
from pathlib import Path

import pandas as pd

import location_resolution as lr
import ner_methods
import week3_build_dataset as w3data
import week3_evaluation as w3
import humaid_loader

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SYNTHETIC_PATH = DATA_DIR / "week3_location_test_synthetic.csv"
REAL_LABELLED_PATH = w3data.HYDRATED_PATH  # HumAID tweets + our manual labels (git-ignored text)
HUMAID_CORPUS_PATH = humaid_loader.OUTPUT_PATH  # full real corpus, NO location labels
OUTPUT_PATH = DATA_DIR / "week4_real_vs_synthetic.csv"

# How many posts to sample from the unlabeled HumAID corpus for the
# extraction-only row (kept small so the experiment stays fast + deterministic).
CORPUS_SAMPLE_SIZE = 300
CORPUS_SAMPLE_SEED = 42

RESULT_FIELDS = [
    "data_origin",
    "dataset_name",
    "label_basis",
    "number_of_posts",
    "location_extraction_rate",
    "exact_or_manual_location_match_rate",
    "ambiguous_location_rate",
    "missing_location_rate",
]


def _pct(n, total):
    return round(100 * n / total, 1) if total else 0.0


def _is_ambiguous(all_locations, text):
    """Extraction-level ambiguity: several places and no disambiguating cue."""
    if len(all_locations) <= 1:
        return False
    _, _, confident = lr.select_primary_location(all_locations, text)
    return not confident


def evaluate_group(texts, expected_labels, extractor):
    """Score one group of posts.

    Parameters
    ----------
    texts : list[str]
    expected_labels : list[str] | None
        Manual ground-truth location labels aligned with ``texts``. ``None``
        means no labels are available (then the match rate is reported as NA).
    extractor : an object with ``.extract(text)`` (from ner_methods).

    Returns a metrics dict.
    """
    total = len(texts)
    extracted = 0
    ambiguous = 0
    exact = 0
    missing = 0
    labelled = 0

    for i, text in enumerate(texts):
        result = extractor.extract(text)
        primary = result["primary_location"]
        all_locations = result["all_locations"]

        if primary:
            extracted += 1
        if _is_ambiguous(all_locations, text):
            ambiguous += 1

        if expected_labels is not None:
            expected = expected_labels[i]
            labelled += 1
            outcome = w3.score_extraction(expected, primary)
            if outcome == "exact":
                exact += 1
            elif outcome == "missing":
                missing += 1

    match_rate = _pct(exact, labelled) if expected_labels is not None else "NA"
    missing_rate = _pct(missing, total) if expected_labels is not None else "NA"

    return {
        "number_of_posts": total,
        "location_extraction_rate": _pct(extracted, total),
        "exact_or_manual_location_match_rate": match_rate,
        "ambiguous_location_rate": _pct(ambiguous, total),
        "missing_location_rate": missing_rate,
    }


def _load_synthetic():
    df = pd.read_csv(SYNTHETIC_PATH).fillna({"expected_location": ""})
    return list(df["text"]), list(df["expected_location"].astype(str))


def _load_real_labelled():
    if not REAL_LABELLED_PATH.exists():
        w3data.build()  # hydrate real HumAID tweet text locally
    df = pd.read_csv(REAL_LABELLED_PATH).fillna({"expected_location": ""})
    return list(df["text"]), list(df["expected_location"].astype(str))


def _load_real_corpus_sample():
    """Return a sample of unlabeled real HumAID posts, or None if unavailable."""
    if not HUMAID_CORPUS_PATH.exists():
        return None
    df = pd.read_csv(HUMAID_CORPUS_PATH)
    if df.empty:
        return None
    n = min(CORPUS_SAMPLE_SIZE, len(df))
    sample = df.sample(n=n, random_state=CORPUS_SAMPLE_SEED)
    return list(sample["original_text"].astype(str))


def run(output_path=OUTPUT_PATH):
    extractor = ner_methods.get_extractor("spacy")
    rows = []

    syn_texts, syn_labels = _load_synthetic()
    rows.append(
        {
            "data_origin": "SYNTHETIC",
            "dataset_name": "project_synthetic",
            "label_basis": "manual_synthetic_labels",
            **evaluate_group(syn_texts, syn_labels, extractor),
        }
    )

    real_texts, real_labels = _load_real_labelled()
    rows.append(
        {
            "data_origin": "REAL_DATASET",
            "dataset_name": "HumAID (labelled subset)",
            "label_basis": "manual_annotations",
            **evaluate_group(real_texts, real_labels, extractor),
        }
    )

    corpus_texts = _load_real_corpus_sample()
    if corpus_texts is not None:
        rows.append(
            {
                "data_origin": "REAL_DATASET",
                "dataset_name": f"HumAID (unlabelled corpus sample n={len(corpus_texts)})",
                "label_basis": "no_location_ground_truth",
                **evaluate_group(corpus_texts, None, extractor),
            }
        )

    df = pd.DataFrame(rows, columns=RESULT_FIELDS)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    results = run()
    print("=== Week 4: real vs synthetic location extraction ===")
    print(results.to_string(index=False))
    print(f"\nWritten to {OUTPUT_PATH}")
    print(
        "\nNote: the HumAID unlabelled-corpus row reports extraction statistics only;"
        "\nits match/missing rates are NA because no ground-truth locations exist."
    )
