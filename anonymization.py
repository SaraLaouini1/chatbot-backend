import os
import json
import requests
from collections import defaultdict

# Endpoints
SERVICE_ADDR    = os.getenv("OLLAMA_SERVICE_ADDRESS")
OLLAMA_CHAT_URL = f"http://{SERVICE_ADDR}/api/chat"
OLLAMA_GEN_URL  = f"http://{SERVICE_ADDR}/api/generate"
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "smollm:360m")

def call_ollama(prompt: str) -> str:
    # Few‑shot examples
    examples = [
        {
            "input":  "My phone is 555-1234 and my SSN is 123-45-6789.",
            "output": '[{"entity":"PHONE","text":"555-1234","start":13,"end":21},'
                      '{"entity":"SSN","text":"123-45-6789","start":32,"end":43}]'
        },
        {
            "input":  "Email me at alice@example.com or call 202-555-0198.",
            "output": '[{"entity":"EMAIL","text":"alice@example.com","start":11,"end":29},'
                      '{"entity":"PHONE","text":"202-555-0198","start":33,"end":46}]'
        },
    ]

    # Build a chat with examples
    messages = [
        {
            "role":    "system",
            "content": (
                "You are a privacy assistant. Extract ALL sensitive entities "
                "and return ONLY a JSON array of {entity,text,start,end}."
            )
        }
    ]
    for ex in examples:
        messages.append({"role": "user",      "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})

    # Finally add the real prompt
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False
    }

    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip()

# (rest of anonymization.py stays unchanged)
def detect_sensitive_entities(text: str) -> list[dict]:
    raw = call_ollama(text)

    # Strip code fences if any
    if raw.startswith("```"):
        parts = raw.split("```", 2)
        if len(parts) == 3:
            raw = parts[2].strip()

    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        out = []
        for ent in data:
            if all(k in ent for k in ("entity","text","start","end")):
                out.append({
                    "entity": str(ent["entity"]),
                    "text":   str(ent["text"]),
                    "start":  int(ent["start"]),
                    "end":    int(ent["end"])
                })
        return out
    except json.JSONDecodeError:
        print("[!] Failed to parse LLM JSON. Raw output:\n", raw)
        return []

def anonymize_text(text: str) -> tuple[str, list[dict]]:
    detected = detect_sensitive_entities(text)
    existing = {}
    counters = defaultdict(int)
    mapping = []
    anonymized = text

    # Replace spans back-to-front so indices stay valid
    for ent in sorted(detected, key=lambda e: e["start"], reverse=True):
        start, end = ent["start"], ent["end"]
        orig = anonymized[start:end]
        key  = (orig, ent["entity"])

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
