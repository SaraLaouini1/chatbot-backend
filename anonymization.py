import requests
import json
import re
from collections import defaultdict
from presidio_analyzer import RecognizerResult

# ---- Configuration ----
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"  # Change as needed

# ---- LLM Call ----
def call_ollama(prompt: str) -> str:
    """Send prompt to local Ollama LLM and return the response text."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        # Ollama response structure: {"message": {"role": ..., "content": ...}}
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        print(f"[!] Ollama Error: {e}")
        return "{}"

# ---- Detection ----
def detect_sensitive_entities(text: str) -> list[dict]:
    """Use LLM to extract sensitive spans as JSON list of {entity, text, start, end}."""
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
    # 1. Detect
    detected = detect_sensitive_entities(text)

    # 2. Prepare mapping containers
    existing = {}                  # key: (orig, entity) -> placeholder
    counters = defaultdict(int)
    mapping: list[dict] = []       # list of {type, original, anonymized}

    # 3. Sort spans backwards and replace
    anonymized = text
    for ent in sorted(detected, key=lambda e: e["start"], reverse=True):
        orig = anonymized[ent["start"]:ent["end"]]
        key = (orig, ent["entity"])
        if key not in existing:
            counters[ent["entity"]] += 1
            placeholder = f"<{ent['entity'].upper()}_{counters[ent['entity']]}>{}"  # e.g. <EMAIL_1>
            existing[key] = placeholder
            mapping.append({
                "type": ent["entity"],
                "original": orig,
                "anonymized": placeholder
            })
        else:
            placeholder = existing[key]
        # Replace in text
        start, end = ent["start"], ent["end"]
        anonymized = anonymized[:start] + placeholder + anonymized[end:]

    return anonymized, mapping
