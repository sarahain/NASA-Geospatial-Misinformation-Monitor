# Data Provenance

Every record in this project carries a `data_origin` that is **exactly one** of
the categories below. They are **never mixed without identifying the source**, and
none is ever relabelled as another.

| `data_origin` | Definition | Examples in this repo | Loader / source |
|---------------|------------|------------------------|-----------------|
| `SYNTHETIC` | Manually created test claims | `data/wildfire_claims.csv`, `data/week3_location_test_synthetic.csv` | written by hand |
| `REAL_DATASET` | Records from established academic datasets | HumAID posts; CrisisMMD posts; curated real tweet references | `humaid_loader.py`, `crisismmd_loader.py`, `wildfire_tweet_refs_loader.py` |
| `REAL_DATASET_HYDRATED` | Historical dataset **tweet IDs** whose text was retrieved through the **official X API** | HumAID IDs rehydrated to text | `x_ingestion.py --mode rehydrate` |
| `LIVE_API` | Posts collected directly from the **official X API** during this project | current disaster posts | `x_ingestion.py --mode search` |

## Definitions (verbatim intent)

- **SYNTHETIC** — manually created test claims. Used only for controlled pipeline
  testing. Never described as tweets.
- **REAL_DATASET** — records from established academic datasets (e.g. HumAID,
  CrisisMMD). The dataset's own text/labels; nothing fabricated.
- **REAL_DATASET_HYDRATED** — historical dataset **IDs** whose text was retrieved
  through the official X API. The IDs come from the academic dataset; the **text**
  is fetched live by official id-lookup and merged with the dataset's event/label
  metadata. Deleted/private IDs get **no** text (never replaced with fake text).
- **LIVE_API** — posts collected directly from the official X API during this
  research project (keyword recent-search).

## Why the distinction matters

- `REAL_DATASET` vs `REAL_DATASET_HYDRATED`: both start from an academic dataset,
  but hydrated records contain text pulled live from X (and so are subject to
  availability — some posts are deleted/private). Keeping them separate makes
  reproducibility and licensing explicit.
- `LIVE_API` is newly collected current content, not from any published dataset.
- `SYNTHETIC` must never be presented as real posts.

## Licensing / what is committed

- **No real post text is committed to Git.** HumAID permits sharing tweet IDs
  only; CrisisMMD/X content is kept out of Git too. All processed text lives under
  the git-ignored `data/processed/` (plus `data/week5_relevance_review_sample.csv`).
- **Tracked, text-free aggregates only**: e.g. `data/week4_real_vs_synthetic.csv`,
  `data/week5_x_collection_summary.csv`, `data/week5_relevance_metrics.csv`,
  `data/week5_x_nasa_ready_summary.csv`.
- **Credentials** (`X_BEARER_TOKEN`, `FIRMS_MAP_KEY`) come only from the
  environment / `.env` (git-ignored); `.env.example` is a token-free template.

## Integrity rules

- A location *word* extracted from a post is never reported as a confirmed
  geolocation. Place metadata from the API is not treated as exact GPS
  coordinates; coordinates are stored only when the API actually provides them.
- Where ground truth does not exist, metrics are reported as `NA` rather than
  fabricated.
- "NASA-ready" means a post *could* be compared against satellite data (has a
  resolvable location + usable timestamp); it does **not** mean verification has
  occurred.
