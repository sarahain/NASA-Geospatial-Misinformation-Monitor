from pathlib import Path

import pandas as pd

import location_resolution as lr
import ner

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLAIMS_PATH = DATA_DIR / "wildfire_claims.csv"
OUTPUT_PATH = DATA_DIR / "location_evaluation.csv"

OUTCOMES = ["correct", "partial", "incorrect", "missing"]


def _match_from_normalized(expected, normalized):
    if not normalized:
        return "missing"
    leading = normalized.split(",")[0]
    return ner.evaluate_match(expected, leading)


def _summarize(method, matches, geocoded, total):
    counts = {o: matches.count(o) for o in OUTCOMES}
    return {
        "method": method,
        "correct": counts["correct"],
        "partial": counts["partial"],
        "incorrect": counts["incorrect"],
        "missing": counts["missing"],
        "exact_match_rate": round(100 * counts["correct"] / total, 1),
        "geocoding_success_rate": None if geocoded is None else round(100 * geocoded / total, 1),
    }


def run_experiment(claims_path=CLAIMS_PATH, output_path=OUTPUT_PATH):
    nlp = ner.load_model()
    claims = pd.read_csv(claims_path)
    total = len(claims)

    a_matches = []
    b_matches, b_geocoded = [], 0
    c_matches, c_geocoded = [], 0

    for _, claim in claims.iterrows():
        expected = claim["expected_location"]
        baseline = ner.pick_location(ner.extract_locations(nlp(claim["text"])))
        a_matches.append(ner.evaluate_match(expected, baseline))

        b = lr.resolve_location(claim["text"], nlp, use_context=False)
        b_matches.append(_match_from_normalized(expected, b["normalized_location"]))
        b_geocoded += 1 if b["latitude"] is not None else 0

        c = lr.resolve_location(claim["text"], nlp, use_context=True)
        c_matches.append(_match_from_normalized(expected, c["normalized_location"]))
        c_geocoded += 1 if c["latitude"] is not None else 0

    rows = [
        _summarize("A_spacy_baseline", a_matches, None, total),
        _summarize("B_spacy_nominatim", b_matches, b_geocoded, total),
        _summarize("C_contextual_resolver", c_matches, c_geocoded, total),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    return summary


if __name__ == "__main__":
    print(run_experiment().to_string(index=False))
