import requests
import json
import re
from collections import defaultdict

# ---- Configuration ----
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"  # Change as needed

# ---- LLM Call ----
def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        print(f"[!] Ollama Error: {e}")
        return "{}"

# ---- Detection ----
def detect_sensitive_entities(text: str) -> list[dict]:
    prompt = (
        f"""
You are a privacy assistant. Extract any personal data from the text below and output a JSON list of objects with keys: entity, text, start, end.
Entities: name, email, phone, address, credit_card, ssn
Text:
{text}
"""
    )
    llm_output = call_ollama(prompt)
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        print("[!] Failed to parse LLM output:", llm_output)
        return []

# ---- Anonymization ----
def anonymize_text(text: str) -> tuple[str, list[dict]]:
    """
    1. Detect sensitive entities with the local LLM
    2. Build mapping of original→placeholder
    3. Replace spans in reverse order
    4. Return anonymized text + mapping list
    """
    detected = detect_sensitive_entities(text)

    existing = {}                    # key: (orig, entity) -> placeholder
    counters = defaultdict(int)
    mapping = []                     # list of {type, original, anonymized}

    anonymized = text
    # Sort descending so that replacing doesn’t shift later spans
    for ent in sorted(detected, key=lambda e: e["start"], reverse=True):
        start, end = ent["start"], ent["end"]
        orig = anonymized[start:end]
        key = (orig, ent["entity"])

        if key not in existing:
            counters[ent["entity"]] += 1
            # FIX: no empty {} in f-string
            placeholder = f"<{ent['entity'].upper()}_{counters[ent['entity']]}>"
            existing[key] = placeholder
            mapping.append({
                "type":       ent["entity"],
                "original":   orig,
                "anonymized": placeholder
            })
        else:
            placeholder = existing[key]

        # Do the replacement
        anonymized = anonymized[:start] + placeholder + anonymized[end:]

    return anonymized, mapping

