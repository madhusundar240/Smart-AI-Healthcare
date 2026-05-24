"""
Season parsing and light reweighting of disease probabilities (India-centric hints).
Not medical fact — only nudges the model toward seasonally plausible patterns.
"""
import re

# Canonical keys stored in session
SEASON_KEYS = ("winter", "summer", "monsoon", "spring", "autumn", "unknown")

SEASON_LABELS = {
    "winter": "Winter (cool / dry months)",
    "summer": "Summer (hot months)",
    "monsoon": "Monsoon / rainy season",
    "spring": "Spring",
    "autumn": "Autumn / fall",
    "unknown": "Not specified",
}

# Multipliers >1.0 nudge probability up; renormalised so top-3 still sum to ~100%.
SEASON_DISEASE_WEIGHTS = {
    "winter": {
        "Flu": 1.38,
        "Common Cold": 1.32,
        "Pneumonia": 1.28,
        "COVID-19": 1.12,
        "Bronchitis": 1.15,
        "Sinusitis": 1.1,
        "Whooping Cough": 1.08,
    },
    "summer": {
        "Heat Stroke": 1.55,
        "Food Poisoning": 1.28,
        "Gastritis": 1.12,
        "Dehydration": 1.1,
        "Migraine": 1.08,
        "Urinary Tract Infection": 1.1,
    },
    "monsoon": {
        "Dengue": 1.48,
        "Malaria": 1.42,
        "Leptospirosis": 1.38,
        "Cholera": 1.25,
        "Typhoid": 1.22,
        "Chikungunya": 1.2,
        "Flu": 1.08,
        "Common Cold": 1.06,
    },
    "spring": {
        "Allergic Reaction": 1.35,
        "Asthma": 1.22,
        "Sinusitis": 1.12,
        "Common Cold": 1.08,
    },
    "autumn": {
        "Flu": 1.22,
        "Common Cold": 1.12,
        "COVID-19": 1.08,
    },
}


def _has_phrase(blob, phrase):
    return re.search(re.escape(phrase.lower()), blob.lower()) is not None


def _has_word(blob, *words):
    """Whole-token match (avoids 'may' matching inside 'maybe')."""
    for w in words:
        if re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", blob, re.I):
            return True
    return False


def parse_season_from_text(text):
    """
    Returns one of SEASON_KEYS, or None if nothing matched.
    Accepts short replies like '1' when options are numbered (caller can map).
    """
    if not text or not str(text).strip():
        return None
    raw = str(text).strip().lower()

    if raw in ("skip", "don't know", "dont know", "no idea", "unknown", "any", "na", "n/a", "-"):
        return "unknown"
    if raw in ("1", "w", "wi"):
        return "winter"
    if raw in ("2", "su"):
        return "summer"
    if raw in ("3", "mon", "rain", "rainy"):
        return "monsoon"
    if raw in ("4", "sp"):
        return "spring"
    if raw in ("5", "au", "fall"):
        return "autumn"

    try:
        from chatbot import detect_language, translate_to_english

        lang = detect_language(text)
        blob = translate_to_english(text, lang).lower() if lang != "en" else raw
    except Exception:
        blob = raw

    blob = re.sub(r"\s+", " ", blob)

    if (
        _has_word(blob, "winter")
        or _has_word(blob, "december", "january", "february")
        or _has_phrase(blob, "cold season")
        or _has_phrase(blob, "peak winter")
    ):
        return "winter"
    if (
        _has_word(blob, "summer", "april", "june")
        or _has_word(blob, "may")
        or _has_phrase(blob, "hot season")
        or _has_phrase(blob, "heat wave")
        or _has_word(blob, "garmi", "grishma")
    ):
        return "summer"
    if (
        _has_word(blob, "monsoon")
        or _has_phrase(blob, "rainy season")
        or _has_word(blob, "rains")
        or _has_phrase(blob, "heavy rain")
        or _has_word(blob, "saavan", "sawan", "mazhai", "barish", "baarish")
        or _has_word(blob, "july", "august", "september")
    ):
        return "monsoon"
    if _has_word(blob, "spring", "march", "vasanth", "basant"):
        return "spring"
    if _has_word(blob, "autumn", "october", "november", "sharad") or _has_word(blob, "fall"):
        return "autumn"

    return None


def apply_season_to_predictions(predictions, season):
    """Renormalise probabilities after per-disease multipliers."""
    if not predictions or not season or season == "unknown":
        return list(predictions)
    weights = SEASON_DISEASE_WEIGHTS.get(season)
    if not weights:
        return list(predictions)
    adjusted = []
    for p in predictions:
        w = float(weights.get(p["disease"], 1.0))
        adjusted.append({"disease": p["disease"], "probability": float(p["probability"]) * w})
    total = sum(x["probability"] for x in adjusted)
    if total <= 0:
        return list(predictions)
    out = [
        {"disease": x["disease"], "probability": round(100.0 * x["probability"] / total, 1)}
        for x in adjusted
    ]
    out.sort(key=lambda x: -x["probability"])
    return out


def client_season_value(raw):
    """
    Normalize a UI/API `season` field. Returns None if the client left season unset (auto).
    Otherwise returns same keys as parse_season_from_text (winter, summer, monsoon, …, unknown).
    """
    if raw is None:
        return None
    t = str(raw).strip().lower()
    if not t or t in ("auto", "ask", "not-set", "default", "none"):
        return None
    return parse_season_from_text(t)


def season_prompt_message():
    return """🌦️ **Which season is it where you are right now?**
Season changes how common some illnesses are (e.g. more flu-like illness in cooler months, more mosquito-borne illness in monsoon in many parts of India).

Reply with **one** of these (English or your language is fine):
• **winter** — cool / dry months (e.g. Dec–Feb)
• **summer** — hot months (e.g. Mar–Jun)
• **monsoon** — rainy season (heavy rains)
• **spring** — Mar–Apr bloom / transition
• **autumn** — fall (e.g. Oct–Nov)

Or type **skip** if you prefer not to say — we will continue without seasonal adjustment.

You can also use the **Season** dropdown above the chat box, or reply **1**–**5** in order: winter, summer, monsoon, spring, autumn."""
