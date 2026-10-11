from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NER_RESULTS_PATH = DATA_DIR / "ner_results.csv"
EVAL_OUTPUT_PATH = DATA_DIR / "ner_evaluation.csv"

OUTCOMES = ["correct", "partial", "incorrect", "missing"]


def evaluate_ner(ner_results_path=NER_RESULTS_PATH, output_path=EVAL_OUTPUT_PATH):
    results = pd.read_csv(ner_results_path).fillna({"extracted_location": ""})
    total = len(results)
    counts = results["ner_match"].value_counts().to_dict()

    rows = []
    for outcome in OUTCOMES:
        count = int(counts.get(outcome, 0))
        rows.append(
            {
                "outcome": outcome,
                "count": count,
                "percentage": round(100 * count / total, 1),
            }
        )

    strict = counts.get("correct", 0)
    lenient = counts.get("correct", 0) + counts.get("partial", 0)
    rows.append({"outcome": "strict_accuracy", "count": strict, "percentage": round(100 * strict / total, 1)})
    rows.append({"outcome": "lenient_success", "count": lenient, "percentage": round(100 * lenient / total, 1)})

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    return summary


if __name__ == "__main__":
    print(evaluate_ner().to_string(index=False))
