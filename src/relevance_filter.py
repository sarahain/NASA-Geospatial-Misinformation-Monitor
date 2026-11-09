"""Transparent, rule-based relevance filter for environmental-disaster posts.

Stage B of the Week 5 collection pipeline. After posts are retrieved from the
official X API, each is classified **locally** (no extra API calls) as:

    RELEVANT            -- a real environmental/disaster signal + supporting context
    POSSIBLY_RELEVANT   -- a hazard is mentioned but with no location/impact/
                           emergency/smoke context
    IRRELEVANT          -- no environmental hazard, or clearly figurative use
                           ("this song is fire", "the team is on fire")

Design principles (per the Week 5 spec)
---------------------------------------
* **Transparent**, not a black box. The score is a sum of inspectable components.
* Keyword presence alone is NOT relevance -- "fire" by itself does not qualify.
* ``relevance_score`` is a 0..1 heuristic, deliberately NOT called a probability
  (it is not calibrated).
* No author popularity / follower counts. No truthfulness judgement -- relevance
  only decides whether a post is *useful* for the research project.

The term lists below are intentionally visible and editable.
"""
import re

# --- Vocabulary (visible + editable) ---------------------------------------
# "Strong" hazard terms clearly denote an environmental hazard on their own.
STRONG_HAZARD_TERMS = [
    "wildfire", "wild fire", "wildfires", "forest fire", "brush fire", "bushfire",
    "wildfire smoke", "smoke plume", "flash flood", "flooding", "floodwaters",
    "drought", "hurricane", "cyclone", "typhoon", "super typhoon", "tornado",
    "mudslide", "landslide", "debris flow", "mudflow", "storm surge",
    "burn scar", "wildfire evacuation",
    # Geophysical (NOT FIRMS-verifiable) -- included at user request.
    "earthquake", "aftershock", "tsunami",
]
# "Weak" terms are ambiguous (figurative-prone) and only count with support.
WEAK_HAZARD_TERMS = ["fire", "smoke", "flood", "blaze", "flames", "burning"]

IMPACT_TERMS = [
    "acre", "acres", "acres burned", "homes", "houses", "structures", "buildings",
    "destroyed", "damaged", "damage", "burned down", "burnt", "containment",
    "contained", "infrastructure", "power outage", "without power", "evacuees",
    "casualties", "fatalities", "killed", "injured", "death toll", "homes damaged",
    "structures destroyed", "scorched", "charred", "submerged", "underwater",
    "stranded", "buried", "washed away", "trapped", "collapsed", "landfall",
    "magnitude", "epicenter", "rubble",
]

EMERGENCY_TERMS = [
    "evacuation", "evacuate", "evacuated", "evacuation order", "evacuation warning",
    "shelter in place", "shelter", "rescue", "road closure", "road closed",
    "firefighters", "fire crews", "first responders", "state of emergency",
    "mandatory evacuation", "red flag warning", "emergency",
]

SMOKE_AIR_TERMS = [
    "smoke", "air quality", "aqi", "unhealthy air", "haze", "smog", "hazardous air",
]

# --- Compact multilingual vocabulary (opt-in, for the small international pass) -
# Covers the highest-volume disaster languages: Spanish (es), Portuguese (pt),
# French (fr), Tagalog (tl), Indonesian/Malay (id). Intentionally small -- just
# the common hazard + impact/emergency words -- so the SAME strict filter can
# keep clearly-stated non-English disaster posts without a separate pipeline.
MULTILINGUAL_HAZARD_TERMS = [
    # es
    "incendio", "incendio forestal", "inundación", "inundaciones", "huracán",
    "terremoto", "sismo", "ciclón", "tifón", "deslizamiento", "tornado",
    # pt
    "incêndio", "inundação", "enchente", "furacão", "ciclone", "deslizamento",
    # fr
    "incendie", "inondation", "ouragan", "séisme", "tremblement de terre",
    "tornade", "typhon", "glissement de terrain",
    # tl
    "sunog", "baha", "bagyo", "lindol",
    # id / ms
    "kebakaran", "banjir", "gempa", "topan", "tanah longsor",
]
MULTILINGUAL_IMPACT_TERMS = [
    # es
    "destruido", "destruidas", "muertos", "heridos", "daños", "casas",
    "desaparecidos", "arrasó",
    # pt
    "destruídas", "mortos", "feridos", "casas", "desaparecidos",
    # fr
    "détruites", "morts", "blessés", "maisons", "disparus",
    # tl
    "nasira", "patay", "sugatan", "bahay",
    # id
    "rumah", "hancur", "tewas", "korban",
]
MULTILINGUAL_EMERGENCY_TERMS = [
    "evacuación", "evacuados", "rescate",          # es
    "evacuação", "resgate",                         # pt
    "évacuation", "évacués", "sauvetage",           # fr
    "paglikas", "sagip",                            # tl
    "evakuasi", "mengungsi",                        # id
]

STRONG_HAZARD_TERMS = STRONG_HAZARD_TERMS + MULTILINGUAL_HAZARD_TERMS
IMPACT_TERMS = IMPACT_TERMS + MULTILINGUAL_IMPACT_TERMS
EMERGENCY_TERMS = EMERGENCY_TERMS + MULTILINGUAL_EMERGENCY_TERMS

# Figurative / non-disaster usages that should be penalised / rejected.
FIGURATIVE_PATTERNS = [
    r"\bthis\s+song\s+is\s+fire\b",
    r"\b(that|this|your|my|his|her|their)\s+(outfit|fit|song|album|track|beat|verse|flow|performance|set|show|food|meal|cooking|look|style|game|drip)\s+(is|was|=)\s+fire\b",
    r"\b(he|she|they|team|he's|she's|they're)\s+(is\s+|are\s+|'s\s+)?on\s+fire\b",
    r"\bstraight\s+fire\b",
    r"\bfire\s+(emoji|emojis)\b",
    r"\blit\s+fire\b",
    r"\bgoes?\s+hard.*fire\b",
    r"\bfire\s+(mixtape|album|track|song|beat)\b",
]

# Detects an explicit NAMED event, e.g. "Palisades Fire", "Camp Fire",
# "Hurricane Maria", "Eaton Fire" -- a strong relevance signal.
EVENT_NAME_RE = re.compile(
    r"\b([A-Z][a-zA-Z]+)\s+(Fire|Fires|Wildfire|Wildfires|Flood|Floods)\b"
    r"|\b(Hurricane|Cyclone|Typhoon|Storm)\s+([A-Z][a-zA-Z]+)\b"
)

# Location signal: "<prep> <Capitalized place>" or administrative keywords.
# Includes a few non-English prepositions (es/pt/fr/tl/id) for the intl pass.
LOCATION_PREP_RE = re.compile(
    r"\b(in|near|at|across|outside|around|north of|south of|east of|west of"
    r"|en|em|à|di|sa|dekat|cerca de|perto de|près de)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)"
)
LOCATION_KEYWORDS = [
    "county", "national park", "national forest", "valley", "canyon", "coast",
    "peninsula", "township", "borough", "province", "state of",
]

# --- Scoring weights (visible + editable) ----------------------------------
W_HAZARD = 0.40
W_LOCATION = 0.25
W_IMPACT = 0.20
W_EMERGENCY = 0.20
W_SMOKE = 0.10
W_FIGURATIVE = 0.60  # penalty magnitude

# Classification thresholds.
RELEVANT_MIN_SCORE = 0.45
# Heavy / "strict quality" mode: higher bar, fewer but cleaner RELEVANT posts.
# In strict mode a post must have hazard + location/event + at least one of
# (impact / emergency / smoke) AND clear this higher score.
# 0.55 lets a STRONG-hazard post with location + one context signal through
# (min realistic score ~0.59) while still demoting weak/figurative-hazard posts.
STRICT_MIN_SCORE = 0.55


def _count_terms(low_text, terms):
    """Count how many of ``terms`` appear in the lowercased text (word-ish).

    Allows a single optional trailing "s" so plurals match too (e.g. the term
    "evacuation" also matches "evacuations", "acre" matches "acres"). This does
    NOT merge distinct words -- "flooding" still won't match "flood" because a
    following letter breaks the trailing word boundary.
    """
    n = 0
    for t in terms:
        # Word-boundary match; phrases match literally; optional plural "s".
        if re.search(r"(?<!\w)" + re.escape(t) + r"s?(?!\w)", low_text):
            n += 1
    return n


def _detect_location(text, low_text):
    if LOCATION_PREP_RE.search(text):
        return True
    return any(k in low_text for k in LOCATION_KEYWORDS)


def _clamp01(x):
    return max(0.0, min(1.0, x))


def classify(text, strict=False):
    """Classify one post's text. Returns a transparent result dict.

    Keys: relevance_label, relevance_score, the component scores, and boolean
    signal flags plus a human-readable ``reason``.

    When ``strict=True`` ("heavy" quality mode), a post is only labelled
    RELEVANT if it has a hazard AND a location/named event AND at least one of
    impact / emergency / smoke context, and clears the higher ``STRICT_MIN_SCORE``.
    Everything that would pass the normal bar but not the strict bar is demoted
    to POSSIBLY_RELEVANT. Default behaviour (``strict=False``) is unchanged.
    """
    text = str(text or "")
    low = text.lower()

    strong = _count_terms(low, STRONG_HAZARD_TERMS)
    weak = _count_terms(low, WEAK_HAZARD_TERMS)
    impact = _count_terms(low, IMPACT_TERMS)
    emergency = _count_terms(low, EMERGENCY_TERMS)
    smoke = _count_terms(low, SMOKE_AIR_TERMS)

    fig_hits = sum(1 for p in FIGURATIVE_PATTERNS if re.search(p, low))

    has_event_name = bool(EVENT_NAME_RE.search(text))
    has_location = _detect_location(text, low)

    # Component scores in [0, 1]. A named event ("Camp Fire", "Hurricane Maria")
    # is itself a strong hazard signal, not merely a location.
    event_signal = 1.0 if has_event_name else 0.0
    hazard_term_score = _clamp01(0.6 * strong + 0.2 * weak + 0.6 * event_signal)
    location_signal = 1.0 if (has_location or has_event_name) else 0.0
    impact_term_score = _clamp01(0.5 * impact)
    emergency_term_score = _clamp01(0.5 * emergency)
    smoke_score = _clamp01(0.5 * smoke)
    figurative_penalty = _clamp01(0.5 * fig_hits)

    relevance_score = _clamp01(
        W_HAZARD * hazard_term_score
        + W_LOCATION * location_signal
        + W_IMPACT * impact_term_score
        + W_EMERGENCY * emergency_term_score
        + W_SMOKE * smoke_score
        - W_FIGURATIVE * figurative_penalty
    )

    # A "real hazard" needs a strong term, OR a weak term backed by real context
    # (impact / emergency / named event / location) -- never a bare weak word.
    has_support = bool(impact or emergency or smoke or has_event_name or has_location)
    has_hazard = strong > 0 or has_event_name or (weak > 0 and has_support)

    # Figurative content with no genuine hazard context is rejected outright.
    figurative_dominates = fig_hits > 0 and strong == 0 and not has_support

    has_damage_terms = impact > 0
    has_emergency_terms = emergency > 0
    has_smoke_air = smoke > 0

    # Strict ("heavy") mode requires a location/event AND concrete context.
    min_score = STRICT_MIN_SCORE if strict else RELEVANT_MIN_SCORE
    strict_context_ok = (
        (has_location or has_event_name)
        and (has_damage_terms or has_emergency_terms or has_smoke_air)
    )
    passes_relevant = has_support and relevance_score >= min_score
    if strict:
        passes_relevant = passes_relevant and strict_context_ok

    if figurative_dominates or not has_hazard:
        label = "IRRELEVANT"
        reason = (
            "figurative/non-disaster use of hazard word"
            if figurative_dominates
            else "no environmental hazard signal"
        )
    elif passes_relevant:
        label = "RELEVANT"
        supports = []
        if has_location or has_event_name:
            supports.append("location/event")
        if has_damage_terms:
            supports.append("impact")
        if has_emergency_terms:
            supports.append("emergency")
        if has_smoke_air:
            supports.append("smoke/air")
        reason = "hazard + " + ", ".join(supports)
    else:
        label = "POSSIBLY_RELEVANT"
        reason = "hazard mentioned but limited disaster context"

    return {
        "relevance_label": label,
        "relevance_score": round(relevance_score, 3),
        "hazard_term_score": round(hazard_term_score, 3),
        "location_signal": round(location_signal, 3),
        "impact_term_score": round(impact_term_score, 3),
        "emergency_term_score": round(emergency_term_score, 3),
        "smoke_score": round(smoke_score, 3),
        "figurative_penalty": round(figurative_penalty, 3),
        "has_hazard": has_hazard,
        "has_location": has_location,
        "has_event_name": has_event_name,
        "has_emergency_terms": has_emergency_terms,
        "has_damage_terms": has_damage_terms,
        "has_smoke_air": has_smoke_air,
        "is_figurative": fig_hits > 0,
        "reason": reason,
    }


def extract_location_hint(text):
    """Return a lightweight location/event string for review display, or "".

    Transparent regex only (named event like "Palisades Fire", or a
    "<prep> <Capitalized>" span). This is a display hint for manual review -- the
    authoritative extraction is the spaCy pipeline in week5_x_ner.py.
    """
    text = str(text or "")
    m = EVENT_NAME_RE.search(text)
    if m:
        return m.group(0)
    m = LOCATION_PREP_RE.search(text)
    if m:
        return m.group(2)
    return ""


def is_research_keepable(result):
    """Whether a classified post may enter the final research subset.

    Keeps RELEVANT posts. POSSIBLY_RELEVANT / IRRELEVANT are stored as candidates
    but do NOT automatically enter the final target set.
    """
    return result["relevance_label"] == "RELEVANT"


if __name__ == "__main__":
    samples = [
        "Mandatory evacuation ordered as the Palisades Fire burns 3000 acres near Los Angeles",
        "Wildfire smoke pushed the AQI to hazardous levels across the Bay Area today",
        "this song is fire \U0001F525\U0001F525",
        "the whole team is on fire tonight what a game",
        "wildfire is crazy",
        "Flash flood warning issued for Ellicott City, homes damaged and roads closed",
    ]
    for s in samples:
        r = classify(s)
        print(f"[{r['relevance_label']:17s} {r['relevance_score']:.2f}] {s[:70]}")
        print(f"    reason: {r['reason']}")
