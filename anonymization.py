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
    # 1️⃣ Try chat API
    chat_payload = {
        "model":    OLLAMA_MODEL,
        "messages": [
            {"role":"system",  "content":"You are a privacy assistant. Return ONLY a JSON array of {entity,text,start,end}."},
            {"role":"user",    "content": prompt}
        ],
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_CHAT_URL, json=chat_payload, timeout=60)
        r.raise_for_status()
        msg = r.json().get("message", {}).get("content", "")
        if msg:
            return msg.strip()
    except Exception:
        pass

    # 2️⃣ Fallback to completion API
    gen_payload = {
        "model":  OLLAMA_MODEL,
        "prompt": (
            "You are a privacy assistant. Extract ALL sensitive entities from the text below "
            "and RETURN ONLY a VALID JSON ARRAY of objects with keys: entity, text, start, end.\n\n"
            "example of output: {"entity": "PHONE", "text": "555-1234", "start": 10, "end": 18}"
            f"{prompt}"
        ),
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_GEN_URL, json=gen_payload, timeout=60)
        r.raise_for_status()
        choices = r.json().get("choices", [])
        if choices:
            return choices[0]["text"].strip()
    except Exception:
        pass

    # 3️⃣ Safe fallback
    return "[]"

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
