import pandas as pd

from src.build_offline_package import quotes_post, scrub_quotes

POST = "Huge wildfire burning near the ridge north of town, smoke everywhere and evacuations ordered now"


def test_quotes_post_detects_long_copied_runs_only():
    assert quotes_post(POST, "User says: " + POST[5:80])
    assert not quotes_post(POST, "wildfire near town")
    assert not quotes_post(POST, "A wildfire with evacuations is reported north of a town; the post is first-hand.")


def test_scrub_quotes_blanks_only_copied_fields():
    df = pd.DataFrame({"text": [POST, POST], "llm_claim": [POST[:70], "wildfire near town"],
                       "llm_reason": ["first-hand report", None]})
    out = scrub_quotes(df)
    assert out["llm_claim"].tolist() == ["[omitted: quotes post text]", "wildfire near town"]
    assert out["llm_reason"].tolist()[0] == "first-hand report"
    assert df["llm_claim"][0] == POST[:70]
