import requests
import json
from collections import defaultdict

# ---- Configuration ----
OLLAMA_URL   = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"  # Change if you’ve pulled another tag

# ---- LLM Call ----
def call_ollama(prompt: str) -> str:
    """
    Send a single-prompt completion to the local Ollama server.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        # Ollama’s /api/generate returns {"id":..., "object":..., "choices":[{"text": "..."}], ...}
        return resp.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[!] Ollama Error: {e}")
        return "{}"

# ---- Detection ----
def detect_sensitive_entities(text: str) -> list[dict]:
    """
    Ask the LLM to locate sensitive spans and return a JSON list of
    { entity, text, start, end }.
    """
    prompt = (
        f"You are a privacy assistant. Extract any personal data from the text below\n"
        f"and output a JSON list of objects with keys: entity, text, start, end.\n"
        f"Entities: name, email, phone, address, credit_card, ssn\n\n"
        f"Text:\n{text}\n\n"
        f"Return ONLY the raw JSON list."
    )
    llm_output = call_ollama(prompt)
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        print("[!] Failed to parse LLM JSON:", llm_output)
        return []

# ---- Anonymization ----
def anonymize_text(text: str) -> tuple[str, list[dict]]:
    """
    1. Detect sensitive entities via local LLM
    2. Build mapping of original→placeholder
    3. Replace spans back-to-front to preserve offsets
    4. Return (anonymized_text, mapping_list)
    """
    detected = detect_sensitive_entities(text)

    existing = {}                    # (orig, entity) → placeholder
    counters = defaultdict(int)
    mapping = []                     # list of {type, original, anonymized}

    anonymized = text
    # Reverse-sort so earlier replacements don’t shift later spans
    for ent in sorted(detected, key=lambda e: e["start"], reverse=True):
        start, end = ent["start"], ent["end"]
        orig = anonymized[start:end]
        key = (orig, ent["entity"])

        if key not in existing:
            counters[ent["entity"]] += 1
            placeholder = f"<{ent['entity'].upper()}_{counters[ent['entity']]}>"
            existing[key] = placeholder
            mapping.append({
                "type":       ent["entity"],
                "original":   orig,
                "anonymized": placeholder
            })
        else:
            placeholder = existing[key]

        anonymized = anonymized[:start] + placeholder + anonymized[end:]

    return anonymized, mapping

