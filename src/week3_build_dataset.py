"""Build the Week 3 *real* evaluation dataset from the HumAID Twitter corpus.

Week 3 was originally evaluated on a labelled synthetic claim set. This script
replaces it with **real social-media claims** drawn from the HumAID dataset
(Alam, Ofli & Imran, 2021 -- https://crisisnlp.qcri.org/humaid_dataset), using
four environmental-hazard events:

    wildfire : canada_wildfires_2016, greece_wildfires_2018
    flood    : srilanka_floods_2017, maryland_floods_2018

Licensing / research integrity
------------------------------
HumAID's terms allow redistribution of **tweet IDs only**, not tweet text
("you can always share the tweet-ids"). Therefore:

* ``data/week3_location_test.csv`` (committed) holds tweet IDs + OUR manual
  location annotations -- NO tweet text.
* ``data/week3_location_test_hydrated.csv`` (git-ignored) is rebuilt locally
  from the HumAID download and DOES contain the real tweet text used by the
  evaluation scripts.

The claim *text* is real and unmodified except for standard tweet normalization
(strip a leading ``RT @user:`` and URLs, collapse whitespace). The
``expected_location`` / ``expected_normalized_location`` fields are OUR
ground-truth annotations, produced by reading each real tweet -- they are not
part of HumAID. Nothing here is fabricated: every row maps to a real, public
tweet id.

Run ``python src/week3_build_dataset.py`` after placing the HumAID
``events_set1`` download under ``data/raw/`` (see README).
"""
import csv
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "events_set1"
PUBLIC_PATH = DATA_DIR / "week3_location_test.csv"            # committed (ids only)
HYDRATED_PATH = DATA_DIR / "week3_location_test_hydrated.csv"  # git-ignored (has text)

DATASET_CITATION = "HumAID (Alam, Ofli & Imran, 2021); https://crisisnlp.qcri.org/humaid_dataset"

# Event -> hazard category, for provenance.
EVENT_CATEGORY = {
    "canada_wildfires_2016": "wildfire",
    "greece_wildfires_2018": "wildfire",
    "srilanka_floods_2017": "flood",
    "maryland_floods_2018": "flood",
}

# --- Objective difficulty rubric -------------------------------------------
# Difficulty is DERIVED from linguistic features of each claim, not assigned by
# hand, so it is reproducible and free of hindsight bias (we do not tune labels
# after seeing model scores). Applied uniformly to all claims:
#
#   hard   : no extractable location (true-negative), OR multiple locations, OR
#            the location appears only inside a #hashtag, OR it is an
#            abbreviation / acronym (e.g. "B.C.", "Wash."), OR a region phrase
#            (e.g. "Northern ...", "south-western").
#   medium : a single place given together with a larger container, e.g.
#            "City, State" (expected is more specific/less specific than given).
#   easy   : a single, plain, in-text place name with none of the above.
REGION_WORDS = {
    "northern", "southern", "south-western", "south-west", "southwest",
    "northwest", "north-west", "northeast", "pacific", "midwest", "central",
}


def classify_difficulty(text, expected_location):
    parts = [p.strip() for p in str(expected_location).split(";") if p.strip()]
    if not parts:
        return "hard"          # no location present -> must avoid false positives
    if len(parts) > 1:
        return "hard"          # multiple locations in one claim
    primary = parts[0]
    low = text.lower()
    first_word = re.sub(r"[^a-z]", "", primary.split()[0].lower())
    if primary.startswith("#"):
        return "hard"          # hashtag-only location
    if first_word:
        plain = re.search(r"(?<!#)\b" + re.escape(first_word) + r"\b", low) is not None
        in_hashtag = ("#" + first_word) in low
        if in_hashtag and not plain:
            return "hard"      # location appears only inside a hashtag
    if "." in primary or (primary.isupper() and len(primary) <= 4):
        return "hard"          # abbreviation / acronym (B.C., Wash., NJ)
    if any(w.lower() in REGION_WORDS for w in primary.replace("-", " ").split()):
        return "hard"          # region phrase, not a single geocodable point
    if "," in primary:
        return "medium"        # "City, State" style
    return "easy"


# --- Manual location annotations -------------------------------------------
# Each tuple: (tweet_id, event, expected_location, expected_normalized_location).
# expected_location uses ';' to list multiple places; an empty string means the
# claim contains NO extractable location (a true-negative). These labels were
# assigned by reading each real tweet; they are our ground truth, not HumAID
# labels. Difficulty is computed from the text by classify_difficulty().
ANNOTATIONS = [
    # ---- Canada wildfires (Fort McMurray, Alberta; some B.C.) ----
    ("733139128284372993", "canada_wildfires_2016", "Ft. Macmurray", "Fort McMurray, Alberta, Canada"),
    ("729279024019722242", "canada_wildfires_2016", "Fort McMurray", "Fort McMurray, Alberta, Canada"),
    ("728609383563862017", "canada_wildfires_2016", "Fort McMurray", "Fort McMurray, Alberta, Canada"),
    ("735888420120219648", "canada_wildfires_2016", "Fort McMurray", "Fort McMurray, Alberta, Canada"),
    ("732284876188045312", "canada_wildfires_2016", "Fort St. John", "Fort St. John, British Columbia, Canada"),
    ("728674229164511232", "canada_wildfires_2016", "Fort McMurray", "Fort McMurray, Alberta, Canada"),
    ("728798158612897792", "canada_wildfires_2016", "Canada", "Canada"),
    ("732239833834229760", "canada_wildfires_2016", "B.C.", "British Columbia, Canada"),
    ("728710371452686336", "canada_wildfires_2016", "British Columbia; Alberta", "British Columbia, Canada"),
    ("730007908411428864", "canada_wildfires_2016", "Fort McMurray", "Fort McMurray, Alberta, Canada"),
    ("732558412710481920", "canada_wildfires_2016", "Fort McMurray; Edmonton", "Fort McMurray, Alberta, Canada"),
    ("734558461783113728", "canada_wildfires_2016", "#yeg", "Edmonton, Alberta, Canada"),
    ("731512686547243010", "canada_wildfires_2016", "Gold Bar, Wash.", "Gold Bar, Washington, United States"),
    ("729822546233712644", "canada_wildfires_2016", "Alberta", "Alberta, Canada"),
    ("728655666349375488", "canada_wildfires_2016", "Fort McMurray", "Fort McMurray, Alberta, Canada"),
    ("730232105137930240", "canada_wildfires_2016", "Fort McMurray", "Fort McMurray, Alberta, Canada"),
    ("729004814865027072", "canada_wildfires_2016", "Alberta", "Alberta, Canada"),
    ("732356254106943489", "canada_wildfires_2016", "#FortMacFire", "Fort McMurray, Alberta, Canada"),
    ("728606357960704006", "canada_wildfires_2016", "Fort McMurray", "Fort McMurray, Alberta, Canada"),
    ("732584373984317440", "canada_wildfires_2016", "Alberta", "Alberta, Canada"),
    ("729143335475027968", "canada_wildfires_2016", "#FortMcMurray", "Fort McMurray, Alberta, Canada"),
    ("734011340403138560", "canada_wildfires_2016", "", ""),
    ("728806631849512960", "canada_wildfires_2016", "Fort McMurray", "Fort McMurray, Alberta, Canada"),
    ("731323477660049408", "canada_wildfires_2016", "", ""),
    ("735159445714411520", "canada_wildfires_2016", "Fort McMurray", "Fort McMurray, Alberta, Canada"),
    ("729340540626472960", "canada_wildfires_2016", "", ""),
    # ---- Greece wildfires (Attica, Athens, Mati; 2018) ----
    ("1024032569023705089", "greece_wildfires_2018", "#Athens", "Athens, Greece"),
    ("1021728450321756165", "greece_wildfires_2018", "Greece", "Greece"),
    ("1021722772849020928", "greece_wildfires_2018", "Greece", "Greece"),
    ("1022027327872675841", "greece_wildfires_2018", "Athens", "Athens, Greece"),
    ("1023346187133440000", "greece_wildfires_2018", "Greece", "Greece"),
    ("1022174982972039172", "greece_wildfires_2018", "Mati; Athens", "Mati, Attica, Greece"),
    ("1029178954777853959", "greece_wildfires_2018", "Chania; Crete", "Chania, Crete, Greece"),
    ("1027109800889524224", "greece_wildfires_2018", "Attica", "Attica, Greece"),
    ("1021741722815143936", "greece_wildfires_2018", "Greece", "Greece"),
    ("1023634394895003648", "greece_wildfires_2018", "Greece", "Greece"),
    ("1022138816033972224", "greece_wildfires_2018", "Greece", "Greece"),
    ("1022233033875881994", "greece_wildfires_2018", "Greece", "Greece"),
    ("1023526815074279424", "greece_wildfires_2018", "Athens", "Athens, Greece"),
    ("1021724736915759104", "greece_wildfires_2018", "Athens", "Athens, Greece"),
    ("1021806419979526145", "greece_wildfires_2018", "Greece; Portugal", "Greece"),
    ("1026012401790054400", "greece_wildfires_2018", "Mati; Greece", "Mati, Attica, Greece"),
    ("1021817205061836802", "greece_wildfires_2018", "Greece; Sweden", "Greece"),
    ("1021776798479249408", "greece_wildfires_2018", "Greece", "Greece"),
    ("1022310663832649728", "greece_wildfires_2018", "Mati; Greece", "Mati, Attica, Greece"),
    ("1021751308775239680", "greece_wildfires_2018", "Greece", "Greece"),
    ("1021753641093550080", "greece_wildfires_2018", "Athens", "Athens, Greece"),
    ("1025330877486850048", "greece_wildfires_2018", "Greece", "Greece"),
    ("1022108745260617730", "greece_wildfires_2018", "Greece", "Greece"),
    ("1021774635682201601", "greece_wildfires_2018", "Greece", "Greece"),
    ("1021770623612342274", "greece_wildfires_2018", "", ""),
    ("1021756551588143110", "greece_wildfires_2018", "", ""),
    # ---- Sri Lanka floods (2017) ----
    ("870261679778562048", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("873397891196440582", "srilanka_floods_2017", "Sri Lanka; Israel", "Sri Lanka"),
    ("870256276869181440", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("879230855549513728", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("870359656379502592", "srilanka_floods_2017", "Galle; Neluwa", "Galle, Sri Lanka"),
    ("878855642890596352", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("879711392550596608", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("877452472125710336", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("870124017734352897", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("870153442664210432", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("870942928482467841", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("873906773777895428", "srilanka_floods_2017", "Sri Lanka; Canada", "Sri Lanka"),
    ("870236474427551744", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("870158705404370945", "srilanka_floods_2017", "Matara", "Matara, Sri Lanka"),
    ("870833926574616577", "srilanka_floods_2017", "Kalutara", "Kalutara, Sri Lanka"),
    ("870172961826852866", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("872080126984339456", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("871148147975659520", "srilanka_floods_2017", "Taiwan", "Taiwan"),
    ("870155426435702784", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("870447164660600833", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("870714879539384320", "srilanka_floods_2017", "Maharagama", "Maharagama, Sri Lanka"),
    ("870848136155025408", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    ("870085868912037888", "srilanka_floods_2017", "Sri Lanka", "Sri Lanka"),
    # ---- Maryland floods (Ellicott City, 2018) ----
    ("1003295658097500160", "maryland_floods_2018", "Ellicott City; Maryland", "Ellicott City, Maryland, United States"),
    ("1001624006485688320", "maryland_floods_2018", "Maryland", "Maryland, United States"),
    ("1001439980995178496", "maryland_floods_2018", "Ellicott, Maryland", "Ellicott City, Maryland, United States"),
    ("1002048883243929600", "maryland_floods_2018", "Ellicott City", "Ellicott City, Maryland, United States"),
    ("1001264739765415936", "maryland_floods_2018", "Maryland", "Maryland, United States"),
    ("1001614473415876608", "maryland_floods_2018", "Ellicott City, Maryland", "Ellicott City, Maryland, United States"),
    ("1002331628369215489", "maryland_floods_2018", "Howard County; Baltimore County", "Howard County, Maryland, United States"),
    ("1001543950845468672", "maryland_floods_2018", "Maryland", "Maryland, United States"),
    ("1001359313305292800", "maryland_floods_2018", "Maryland", "Maryland, United States"),
    ("1001197004247183361", "maryland_floods_2018", "Ellicott City, Maryland", "Ellicott City, Maryland, United States"),
    ("1003442050652626945", "maryland_floods_2018", "Cheverly, Maryland", "Cheverly, Maryland, United States"),
    ("1001177012512350209", "maryland_floods_2018", "Elliott City", "Ellicott City, Maryland, United States"),
    ("1001192363719618561", "maryland_floods_2018", "Elicott City, Maryland", "Ellicott City, Maryland, United States"),
    ("1001228400697249792", "maryland_floods_2018", "Ellicott City, Maryland", "Ellicott City, Maryland, United States"),
    ("1002511281511776256", "maryland_floods_2018", "Maryland", "Maryland, United States"),
    ("1001203964648607747", "maryland_floods_2018", "Maryland", "Maryland, United States"),
    ("1001528621641723904", "maryland_floods_2018", "Ellicott City; Maryland", "Ellicott City, Maryland, United States"),
    ("1001260095223758850", "maryland_floods_2018", "Ellicott City, Maryland", "Ellicott City, Maryland, United States"),
    ("1003294137335066624", "maryland_floods_2018", "Maryland; Virginia; Delaware", "Maryland, United States"),
    ("1001183184229359617", "maryland_floods_2018", "Maryland", "Maryland, United States"),
    ("1001456447245336576", "maryland_floods_2018", "Ellicott City, Maryland", "Ellicott City, Maryland, United States"),
    ("1002052137835261954", "maryland_floods_2018", "Ellicott City; Maryland", "Ellicott City, Maryland, United States"),
    ("1001915701546881024", "maryland_floods_2018", "Ellicott City; Maryland", "Ellicott City, Maryland, United States"),
    ("1001443716320579585", "maryland_floods_2018", "", ""),
    ("1003706990869311488", "maryland_floods_2018", "", ""),
]

PUBLIC_FIELDS = [
    "claim_id", "tweet_id", "category", "expected_location",
    "expected_normalized_location", "difficulty", "source_type", "dataset",
]
HYDRATED_FIELDS = PUBLIC_FIELDS[:2] + ["text"] + PUBLIC_FIELDS[2:]


def clean_text(text):
    """Standard, minimal tweet normalization (keeps the real claim content)."""
    text = re.sub(r"^RT @\w+:\s*", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_tweet_texts():
    """Map tweet_id -> cleaned text from the local HumAID TSVs under data/raw."""
    if not RAW_DIR.exists():
        raise SystemExit(
            f"HumAID corpus not found at {RAW_DIR}.\n"
            "Download 'Event wise dataset (set1)' from "
            "https://crisisnlp.qcri.org/humaid_dataset and extract it so that "
            f"{RAW_DIR} exists, then re-run this script."
        )
    texts = {}
    for tsv in RAW_DIR.rglob("*.tsv"):
        with open(tsv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                tid = row.get("tweet_id")
                if tid and tid not in texts:
                    texts[tid] = clean_text(row.get("tweet_text", ""))
    return texts


def build():
    texts = load_tweet_texts()

    missing = [tid for tid, *_ in ANNOTATIONS if tid not in texts]
    if missing:
        raise SystemExit(f"{len(missing)} annotated tweet ids not found in corpus: {missing[:5]}")

    public_rows, hydrated_rows = [], []
    for i, (tid, event, exp, exp_norm) in enumerate(ANNOTATIONS, start=1):
        text = texts[tid]
        base = {
            "claim_id": i,
            "tweet_id": tid,
            "category": EVENT_CATEGORY[event],
            "expected_location": exp,
            "expected_normalized_location": exp_norm,
            "difficulty": classify_difficulty(text, exp),
            "source_type": "real",
            "dataset": f"{DATASET_CITATION} [{event}]",
        }
        public_rows.append(base)
        hydrated_rows.append({**base, "text": text})

    _write(PUBLIC_PATH, PUBLIC_FIELDS, public_rows)
    _write(HYDRATED_PATH, HYDRATED_FIELDS, hydrated_rows)
    return public_rows, hydrated_rows


def _write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    public_rows, hydrated_rows = build()
    cats = {}
    diffs = {}
    tns = 0
    for r in public_rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
        diffs[r["difficulty"]] = diffs.get(r["difficulty"], 0) + 1
        if not r["expected_location"]:
            tns += 1
    print(f"Built {len(public_rows)} real claims")
    print(f"  committed (ids only): {PUBLIC_PATH.name}")
    print(f"  local (with text)   : {HYDRATED_PATH.name}")
    print(f"  categories : {cats}")
    print(f"  difficulty : {diffs}")
    print(f"  no-location true-negatives: {tns}")
