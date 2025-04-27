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
        "You are a privacy assistant. Extract ALL sensitive entities from this text.\n"
        "RETURN ONLY A VALID JSON ARRAY of objects with EXACTLY these keys:\n"
        "- entity (string, type: PERSON, EMAIL, PHONE, CREDIT_CARD, etc.)\n"
        "- text (exact matched text)\n"
        "- start (integer start index)\n"
        "- end (integer end index)\n"
        "FORMAT EXAMPLE:\n"
        "[{\"entity\": \"EMAIL\", \"text\": \"user@example.com\", \"start\": 42, \"end\": 58}]\n"
        "DIRECTIVES:\n"
        "- No explanations\n"
        "- No markdown/code formatting\n"
        "- Validate JSON syntax\n"
        "- Only include complete, verified matches\n"
        f"TEXT TO ANALYZE:\n{text}"
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
        return data.get("response", "").strip()
    except Exception as e:
        print(f"[!] Ollama Error contacting {OLLAMA_URL}: {e}")
        return "[]"

def detect_sensitive_entities(text: str) -> list[dict]:
    """
    Add JSON sanitization and validation
    """
    llm_output = call_ollama(text)
    
    # Attempt to extract JSON from markdown code blocks
    if llm_output.startswith("```json"):
        llm_output = llm_output[7:-3].strip()  # Remove ```json and trailing ```
    
    try:
        parsed = json.loads(llm_output)
        # Validate structure
        if not isinstance(parsed, list):
            return []
            
        valid_entries = []
        for entry in parsed:
            if all(key in entry for key in ("entity", "text", "start", "end")):
                valid_entries.append({
                    "entity": str(entry["entity"]),
                    "text": str(entry["text"]),
                    "start": int(entry["start"]),
                    "end": int(entry["end"])
                })
        return valid_entries
        
    except json.JSONDecodeError:
        print(f"[!] Failed to parse LLM JSON. Raw output:\n{llm_output[:200]}...")
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
