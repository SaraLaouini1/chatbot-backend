# anonymization.py

import json
import re
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── 1. Load your local LLM ─────────────────────────────────────────────────────

MODEL_NAME = "TheBloke/Llama-2-7B-GGUF"   # or whichever local checkpoint you have
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if DEVICE=="cuda" else None,
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
).to(DEVICE)


# ─── 2. A helper that asks the LLM to return ALL spans of sensitive data ────────

def detect_sensitive_spans(text: str):
    """
    Prompts the local LLM to find every PII or legal reference span in `text`.
    Returns a list of dicts: [{"type": TYPE, "start": int, "end": int}, …]
    """
    instruction = (
        "You are a data-privacy assistant. Given the user’s text, "
        "identify every span of **sensitive data**—including:\n"
        " • Emails\n"
        " • Phone numbers\n"
        " • IP addresses\n"
        " • Credit card numbers\n"
        " • SSNs or national IDs\n"
        " • Person names, Orgs, Locations, Dates\n"
        " • Clause references (e.g. “Clause 5.3”)\n"
        " • Case numbers (e.g. “Case No. 2021-CR-04567”)\n\n"
        "Return ONLY a JSON array of objects with keys "
        "`type` (string), `start` (int), `end` (int).  "
        "Do NOT return any other text.\n\n"
        f"Text:\n'''{text}'''"
    )

    # Tokenize + generate
    inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the JSON array from the output
    try:
        # assumes the model outputs something like: ...[ {...}, {...} ]...
        json_str = generated[generated.index("["): generated.rindex("]")+1]
        spans    = json.loads(json_str)
        # Basic validation
        clean = []
        for s in spans:
            if (
                isinstance(s.get("type"), str)
                and isinstance(s.get("start"), int)
                and isinstance(s.get("end"),   int)
                and 0 <= s["start"] < s["end"] <= len(text)
            ):
                clean.append(s)
        return clean
    except Exception:
        return []


# ─── 3. The anonymization function ───────────────────────────────────────────────

def anonymize_text(text: str):
    """
    1. Calls detect_sensitive_spans(text) via your local LLM.
    2. Replaces each span with <TYPE_n> tokens (new counter per type).
    3. Returns (anonymized_text, mapping_list).
    """
    spans = detect_sensitive_spans(text)

    # Sort spans in reverse order so earlier replacements don't shift indices
    spans = sorted(spans, key=lambda x: x["start"], reverse=True)

    existing, counters, mapping = {}, defaultdict(int), []
    out = text

    for span in spans:
        typ, start, end = span["type"], span["start"], span["end"]
        orig = out[start:end]
        key  = (orig, typ)

        if key not in existing:
            counters[typ] += 1
            token = f"<{typ}_{counters[typ]}>"
            existing[key] = token
            mapping.append({
                "type":       typ,
                "original":   orig,
                "anonymized": token
            })
        else:
            token = existing[key]

        # Splice in the placeholder
        out = out[:start] + token + out[end:]

    return out, mapping
