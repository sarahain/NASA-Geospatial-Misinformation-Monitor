from pathlib import Path

import pandas as pd
import spacy

LOCATION_LABELS = {"GPE", "LOC", "FAC"}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLAIMS_PATH = DATA_DIR / "wildfire_claims.csv"
NER_OUTPUT_PATH = DATA_DIR / "ner_results.csv"


def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise SystemExit(
            "spaCy model 'en_core_web_sm' is not installed.\n"
            "Run: python -m spacy download en_core_web_sm"
        )


def extract_locations(doc):
    return [ent.text for ent in doc.ents if ent.label_ in LOCATION_LABELS]


def pick_location(entities):
    if not entities:
        return ""
    return entities[0]


def normalize(text):
    return " ".join(text.lower().replace(",", " ").split())


def evaluate_match(expected, extracted):
    if not extracted:
        return "missing"
    expected_n = normalize(expected)
    extracted_n = normalize(extracted)
    if extracted_n == expected_n:
        return "correct"
    if extracted_n in expected_n or expected_n in extracted_n:
        return "partial"
    return "incorrect"


def run_ner(claims_path=CLAIMS_PATH, output_path=NER_OUTPUT_PATH):
    nlp = load_model()
    claims = pd.read_csv(claims_path)

    rows = []
    for _, claim in claims.iterrows():
        doc = nlp(claim["text"])
        entities = extract_locations(doc)
        extracted = pick_location(entities)
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "text": claim["text"],
                "expected_location": claim["expected_location"],
                "extracted_location": extracted,
                "all_location_entities": "; ".join(entities),
                "ner_match": evaluate_match(claim["expected_location"], extracted),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    df = run_ner()
    print(df.to_string(index=False))
