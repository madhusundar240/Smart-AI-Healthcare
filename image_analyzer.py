"""
Vision-based symptom hints from user photos.

Order: cloud APIs (optional keys) → Ollama (local) → **built-in BLIP caption**
(BLIP needs `pip install torch transformers pillow`; first run downloads ~1 GB).
Output maps to model symptom codes — not a medical diagnosis.
"""
import base64
import json
import os
import re

import joblib
import requests

from local_image_caption import caption_image_bytes

_MODEL_SYMPTOMS = None


def _allowed_symptoms():
    global _MODEL_SYMPTOMS
    if _MODEL_SYMPTOMS is None:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "models", "symptoms_list.pkl")
        _MODEL_SYMPTOMS = set(joblib.load(path))
    return _MODEL_SYMPTOMS


def _parse_json_object(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _validate_symptoms(raw_list):
    allowed = _allowed_symptoms()
    out = []
    if not isinstance(raw_list, list):
        return out
    for item in raw_list:
        if isinstance(item, str) and item.strip() in allowed:
            out.append(item.strip())
    return list(dict.fromkeys(out))


def _vision_prompt():
    codes = sorted(_allowed_symptoms())
    codes_csv = ",".join(codes)
    return f"""You are assisting a triage-style symptom checker (NOT a doctor).
Look at the image. Infer only signs that are reasonably visible or clearly implied (e.g. thermometer reading, rash photo, eye redness, swelling, jaundice tint).
Return STRICT JSON only, no markdown, with this shape:
{{"symptoms": ["<codes>"],"notes":"<one short sentence>"}}

Rules:
- "symptoms" must be a subset of these exact underscore codes (comma-separated reference): {codes_csv}
- Use [] if the image is not medical, too blurry, or nothing can be inferred.
- Never invent codes outside the list. Use the closest allowed code (e.g. use "rash" or "skin_rash" if both exist — prefer the one from the list exactly).
- Do not output probabilities or diagnoses; only observable clues as codes.
"""


def _symptoms_from_visual_caption(caption, allowed):
    """Map generic BLIP-style captions to dataset symptom codes (best-effort)."""
    if not caption:
        return []
    low = caption.lower()
    found = []

    def add(code):
        if code in allowed and code not in found:
            found.append(code)

    pairs = [
        ("jaundice", "jaundice"),
        ("yellow skin", "yellow_skin"),
        ("yellow eyes", "yellow_eyes"),
        ("pale skin", "pale_skin"),
        ("swollen", "swelling"),
        ("swelling", "swelling"),
        ("joint", "joint_pain"),
        ("eye pain", "eye_pain"),
        ("red eye", "red_eyes"),
        ("bloodshot", "red_eyes"),
        ("nausea", "nausea"),
        ("vomit", "vomiting"),
        ("diarrhea", "diarrhea"),
        ("cough", "cough"),
        ("sneeze", "sneezing"),
        ("headache", "headache"),
        ("dizzy", "dizziness"),
        ("fatigue", "fatigue"),
        ("tired", "fatigue"),
        ("thermometer", "fever"),
        ("temperature", "fever"),
    ]
    for phrase, code in pairs:
        if phrase in low:
            add(code)

    if re.search(
        r"\b(rash|hives|blister|lesion|eczema|psoriasis|spots on skin|skin rash|measles|chickenpox|pimple)\b",
        low,
    ):
        if "skin_rash" in allowed:
            add("skin_rash")
        elif "rash" in allowed:
            add("rash")
    if re.search(r"\b(bandage|bruise|wound|cut|bleed)\b", low):
        add("bleeding")
    return found


def _try_local_blip(image_bytes):
    """Built-in caption → symptoms (no API keys)."""
    cap = caption_image_bytes(image_bytes)
    if not cap:
        return None
    try:
        from chatbot import extract_symptoms_from_text
    except Exception:
        extract_symptoms_from_text = None
    allowed = _allowed_symptoms()
    syms = []
    if extract_symptoms_from_text:
        syms.extend(extract_symptoms_from_text(cap))
    syms.extend(_symptoms_from_visual_caption(cap, allowed))
    syms = [s for s in dict.fromkeys(syms) if s in allowed]
    return {"symptoms": syms, "notes": cap.strip(), "provider": "local_blip"}


def _call_anthropic(image_bytes, media_type):
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("ANTHROPIC_VISION_MODEL", "claude-3-5-sonnet-20241022")
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    body = {
        "model": model,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type or "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": _vision_prompt()},
                ],
            }
        ],
    }
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    parts = data.get("content") or []
    texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
    return "".join(texts)


def _call_openai(image_bytes, media_type):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    mime = media_type or "image/jpeg"
    body = {
        "model": model,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _vision_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
    }
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


def _call_gemini(image_bytes, media_type):
    key = (
        os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )
    if not key:
        return None
    model = os.environ.get("GEMINI_VISION_MODEL", "gemini-1.5-flash")
    mime = media_type or "image/jpeg"
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {
        "contents": [
            {
                "parts": [
                    {"text": _vision_prompt()},
                    {"inline_data": {"mime_type": mime, "data": b64}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }
    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts)


def _call_ollama(image_bytes, media_type):
    if os.environ.get("OLLAMA_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        return None
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_VISION_MODEL", "llava").strip() or "llava"
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    url = f"{base}/api/chat"
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": _vision_prompt(),
                "images": [b64],
            }
        ],
        "stream": False,
    }
    r = requests.post(url, json=body, timeout=90)
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip() or None


def analyze_image_for_symptoms(image_bytes, content_type="image/jpeg"):
    """
    Returns dict:
      ok True: { ok, symptoms, notes, provider }
      ok False: { ok, error, hint }
    """
    if not image_bytes or len(image_bytes) < 32:
        return {"ok": False, "error": "empty_image", "hint": "Please choose a valid image file."}

    ct = (content_type or "image/jpeg").split(";")[0].strip()
    if ct not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        ct = "image/jpeg"

    text = None
    provider = None
    for name, fn in (
        ("anthropic", _call_anthropic),
        ("openai", _call_openai),
        ("google", _call_gemini),
        ("ollama", _call_ollama),
    ):
        try:
            text = fn(image_bytes, ct)
            if text:
                provider = name
                break
        except Exception:
            text = None
            continue

    if text:
        try:
            obj = _parse_json_object(text)
        except Exception:
            fb = _try_local_blip(image_bytes)
            if fb:
                return {"ok": True, **fb}
            return {
                "ok": False,
                "error": "bad_model_output",
                "hint": "The vision model returned unreadable data. Try another photo or type your symptoms.",
                "raw_preview": text[:400],
            }
        symptoms = _validate_symptoms(obj.get("symptoms"))
        notes = obj.get("notes") if isinstance(obj.get("notes"), str) else ""
        return {"ok": True, "symptoms": symptoms, "notes": notes.strip(), "provider": provider}

    fb = _try_local_blip(image_bytes)
    if fb:
        return {"ok": True, **fb}

    return {
        "ok": False,
        "error": "no_vision_provider",
        "hint": (
            "Built-in image caption (BLIP) did not run. From the project folder run:\n"
            "  pip install torch transformers pillow\n"
            "Then restart the app — the first image may take a minute while the model downloads (~1 GB).\n\n"
            "Optional: Ollama (https://ollama.com, ollama pull llava) or cloud keys "
            "(ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY). "
            "Set LOCAL_BLIP_DISABLE=1 to skip BLIP. You can still type or speak symptoms."
        ),
    }
