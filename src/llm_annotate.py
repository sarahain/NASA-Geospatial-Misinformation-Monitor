"""Local-LLM annotation of collected X posts (Ollama).

Stage between collection (``x_ingestion.py``) and satellite verification
(``verify_tweets.py``). It asks a **local** LLM served by Ollama to turn each raw
post into a small, structured annotation that the geocode -> FIRMS -> verify
pipeline can act on:

    is_hazard_claim : is the post actually asserting a real-world hazard event?
    hazard_type     : wildfire / flood / tornado / hurricane / typhoon /
                      mudslide / earthquake / severe_weather / other / none
    location        : the place the event is claimed to be (free text, "" if none)
    claim           : the single checkable factual claim, in plain words
    stance          : asserts_event / denies_event / opinion / unrelated
    misinfo_risk    : low / medium / high  (model's own rough judgement)
    confidence      : 0..1
    reason          : one short sentence

Design
------
* **Local only.** Uses Ollama's HTTP API (default http://localhost:11434). No
  cloud provider, no API key, no per-token cost.
* **No network at import.** All HTTP happens inside the annotation calls.
* **Offline-friendly / testable.** ``annotate_text`` accepts a ``responder``
  callable so unit tests can inject fake model output without a running server.
* **Resumable.** Already-annotated post_ids in the output are skipped on re-run.

Nothing here spends X API credits -- it only reads a local CSV and talks to the
local model.
"""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_INPUT = PROCESSED_DIR / "x_relevant_merged.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "x_llm_annotated.csv"

OLLAMA_HOST_ENV = "OLLAMA_HOST"
OLLAMA_MODEL_ENV = "OLLAMA_MODEL"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"

HAZARD_TYPES = {
    "wildfire", "flood", "tornado", "hurricane", "typhoon", "mudslide",
    "earthquake", "severe_weather", "other", "none",
}
STANCES = {"asserts_event", "denies_event", "opinion", "unrelated"}
RISKS = {"low", "medium", "high"}

# Columns copied straight through from the input, then the llm_* annotations.
PASSTHROUGH_FIELDS = [
    "post_id", "text", "created_at", "lang", "geo_coordinates",
    "has_media", "media_types",
]
ANNOTATION_FIELDS = [
    "llm_is_hazard_claim", "llm_hazard_type", "llm_location", "llm_claim",
    "llm_stance", "llm_misinfo_risk", "llm_confidence", "llm_reason",
    "llm_status",
]
OUTPUT_FIELDS = PASSTHROUGH_FIELDS + ANNOTATION_FIELDS

SYSTEM_PROMPT = (
    "You are a careful annotator for a disaster/misinformation research project. "
    "Given one social-media post, extract a STRICT JSON object describing whether "
    "it asserts a real-world environmental/disaster hazard event and where. "
    "Do not invent a location that is not in the text. Only output JSON."
)

JSON_INSTRUCTIONS = (
    "Return ONLY a JSON object with exactly these keys:\n"
    '  "is_hazard_claim": boolean,\n'
    '  "hazard_type": one of ["wildfire","flood","tornado","hurricane","typhoon",'
    '"mudslide","earthquake","severe_weather","other","none"],\n'
    '  "location": string (place named in the post, else ""),\n'
    '  "claim": string (the single checkable claim in plain words, else ""),\n'
    '  "stance": one of ["asserts_event","denies_event","opinion","unrelated"],\n'
    '  "misinfo_risk": one of ["low","medium","high"],\n'
    '  "confidence": number between 0 and 1,\n'
    '  "reason": one short sentence.\n'
)


class LLMAnnotateError(RuntimeError):
    """Configuration / server problems the caller can report cleanly."""


def _require_requests():
    try:
        import requests  # noqa: F401
    except Exception as exc:
        raise LLMAnnotateError(f"the 'requests' package is required: {exc}") from exc
    import requests

    return requests


def get_host():
    return os.environ.get(OLLAMA_HOST_ENV, "").strip() or DEFAULT_HOST


def get_model():
    return os.environ.get(OLLAMA_MODEL_ENV, "").strip() or DEFAULT_MODEL


def build_prompt(text):
    return f"{JSON_INSTRUCTIONS}\nPOST:\n{text}\n"


def _ollama_chat(prompt, model, host, session=None):
    """Call the local Ollama chat API and return the raw text content."""
    requests = _require_requests()
    session = session or requests.Session()
    url = f"{host.rstrip('/')}/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    try:
        resp = session.post(url, json=body, timeout=600)
    except Exception as exc:
        raise LLMAnnotateError(
            f"could not reach Ollama at {host}. Is it running? "
            f"Start it with `ollama serve` and `ollama pull {model}`. ({exc})"
        ) from exc
    if resp.status_code != 200:
        raise LLMAnnotateError(
            f"Ollama returned HTTP {resp.status_code}. Check the model name "
            f"('{model}') is pulled: `ollama pull {model}`."
        )
    data = resp.json()
    # Ollama /api/chat -> {"message": {"content": "..."}}
    return (data.get("message", {}) or {}).get("content", "") or ""


def _coerce(raw_obj):
    """Validate / normalize the model's JSON into the annotation schema."""
    obj = raw_obj if isinstance(raw_obj, dict) else {}

    def _str(v):
        return "" if v is None else str(v).strip()

    hazard = _str(obj.get("hazard_type")).lower()
    if hazard not in HAZARD_TYPES:
        hazard = "other" if obj.get("is_hazard_claim") else "none"
    stance = _str(obj.get("stance")).lower()
    if stance not in STANCES:
        stance = "unrelated"
    risk = _str(obj.get("misinfo_risk")).lower()
    if risk not in RISKS:
        risk = "low"
    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "llm_is_hazard_claim": bool(obj.get("is_hazard_claim")),
        "llm_hazard_type": hazard,
        "llm_location": _str(obj.get("location")),
        "llm_claim": _str(obj.get("claim")),
        "llm_stance": stance,
        "llm_misinfo_risk": risk,
        "llm_confidence": round(conf, 3),
        "llm_reason": _str(obj.get("reason")),
        "llm_status": "ok",
    }


def _empty_annotation(status):
    return {
        "llm_is_hazard_claim": "",
        "llm_hazard_type": "",
        "llm_location": "",
        "llm_claim": "",
        "llm_stance": "",
        "llm_misinfo_risk": "",
        "llm_confidence": "",
        "llm_reason": "",
        "llm_status": status,
    }


def annotate_text(text, model=None, host=None, session=None, responder=None):
    """Annotate one post's text -> annotation dict.

    ``responder`` (callable ``prompt -> raw_json_str``) lets tests bypass the
    network. When absent, the local Ollama server is used.
    """
    text = str(text or "").strip()
    if not text:
        return _empty_annotation("empty_text")
    model = model or get_model()
    host = host or get_host()
    prompt = build_prompt(text)
    raw = responder(prompt) if responder is not None else _ollama_chat(prompt, model, host, session)
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return _empty_annotation("unparseable_json")
    return _coerce(parsed)


def _read_rows(path):
    path = Path(path)
    if not path.exists():
        raise LLMAnnotateError(f"input file not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _already_done(output_path):
    out = Path(output_path)
    if not out.exists():
        return set()
    with open(out, newline="", encoding="utf-8") as fh:
        return {r.get("post_id", "") for r in csv.DictReader(fh)}


def annotate_file(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, model=None,
                  host=None, limit=None, only_relevant=True, session=None,
                  responder=None):
    """Annotate posts from ``input_path`` -> ``output_path`` (resumable)."""
    rows = _read_rows(input_path)
    if only_relevant:
        rows = [r for r in rows if r.get("relevance_label", "RELEVANT") == "RELEVANT"]
    done = _already_done(output_path)
    rows = [r for r in rows if r.get("post_id", "") not in done]
    if limit is not None:
        rows = rows[:limit]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    n_ok = 0
    with open(out, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for r in rows:
            ann = annotate_text(r.get("text", ""), model=model, host=host,
                                session=session, responder=responder)
            record = {k: r.get(k, "") for k in PASSTHROUGH_FIELDS}
            record.update(ann)
            writer.writerow(record)
            fh.flush()
            if ann["llm_status"] == "ok":
                n_ok += 1
    return {"annotated": len(rows), "ok": n_ok, "output": str(out)}


def ping(host=None, session=None):
    """Return True if a local Ollama server answers at ``host``."""
    requests = _require_requests()
    session = session or requests.Session()
    host = host or get_host()
    try:
        resp = session.get(f"{host.rstrip('/')}/api/tags", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Annotate collected X posts with a local LLM (Ollama).")
    p.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV of collected posts.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output annotated CSV.")
    p.add_argument("--model", default=None, help=f"Ollama model (default ${OLLAMA_MODEL_ENV} or {DEFAULT_MODEL}).")
    p.add_argument("--host", default=None, help=f"Ollama host (default ${OLLAMA_HOST_ENV} or {DEFAULT_HOST}).")
    p.add_argument("--limit", type=int, default=None, help="Annotate at most N posts.")
    p.add_argument("--all-labels", action="store_true",
                   help="Annotate every row, not just relevance_label==RELEVANT.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config + check the local model is reachable; annotate nothing.")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    model = args.model or get_model()
    host = args.host or get_host()

    if args.dry_run:
        reachable = ping(host=host)
        print("=== llm_annotate DRY RUN (no annotation performed) ===")
        print(f"input            : {args.input}")
        print(f"output           : {args.output}")
        print(f"ollama host      : {host}")
        print(f"ollama model     : {model}")
        print(f"only RELEVANT    : {not args.all_labels}")
        print(f"server reachable : {'YES' if reachable else 'NO -> start `ollama serve`'}")
        print(f"model note       : ensure it is pulled: `ollama pull {model}`")
        return 0 if reachable else 2

    result = annotate_file(
        input_path=args.input, output_path=args.output, model=model, host=host,
        limit=args.limit, only_relevant=not args.all_labels,
    )
    print("\n--- llm_annotate result ---")
    print(f"posts annotated : {result['annotated']}")
    print(f"ok              : {result['ok']}")
    print(f"output          : {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
