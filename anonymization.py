import requests
import json
import os
from collections import defaultdict

# Build the Ollama completions URL
SERVICE_ADDR = os.getenv("OLLAMA_SERVICE_ADDRESS")
OLLAMA_URL   = f"http://{SERVICE_ADDR}/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "smollm:360m")

def call_ollama(text: str) -> str:
    """
    Send a completion request to the Ollama /api/generate endpoint with
    a strict instruction to output ONLY a JSON list of entities.
    """
    instruction = (
        "You are a privacy assistant. Extract all sensitive data from the text "
        "below and RETURN ONLY a JSON array of objects with keys: entity, text, start, end. "
        "Do NOT include any explanation, headings, code fences, or anything else—"
        "just the raw JSON array.\n\n"
        f"Text:\n{text}"
    )
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": instruction,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Ollama /api/generate uses "choices"[{"text": ...}]
        return data["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[!] Ollama Error contacting {OLLAMA_URL}: {e}")
        # Guaranteed valid JSON fallback
        return "[]"

def detect_sensitive_entities(text: str) -> list[dict]:
    """
    Ask the LLM to locate sensitive spans and return a JSON list of
    { entity, text, start, end }.
    """
    llm_output = call_ollama(text)
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        print("[!] Failed to parse LLM JSON:", llm_output)
        return []

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
