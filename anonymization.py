# anonymization.py

import os
import json
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

# ─── 1. Configuration ───────────────────────────────────────────────────────────

# HF repo name of your local checkpoint
MODEL_NAME = "TheBloke/Llama-2-7B-GGUF"
# Where to store it on disk
MODEL_DIR  = os.getenv("MODEL_DIR", "./models/Llama-2-7B-GGUF")
# HF token (set in Render’s env)
HF_TOKEN   = os.getenv("HF_TOKEN", None)

# Device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy holders
_tokenizer = None
_model     = None


# ─── 2. Lazy-loader + downloader ────────────────────────────────────────────────

def _ensure_local_llm():
    global _tokenizer, _model

    # already loaded?
    if _tokenizer and _model:
        return

    # 2.1 Download if missing
    if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
        if not HF_TOKEN:
            print("[anonymization] WARN: no HF_TOKEN, can't download model")
        else:
            try:
                print(f"[anonymization] downloading model to {MODEL_DIR} …")
                snapshot_download(
                    repo_id=MODEL_NAME,
                    local_dir=MODEL_DIR,
                    use_auth_token=HF_TOKEN,
                    resume_download=True
                )
                print("[anonymization] download complete")
            except Exception as e:
                print(f"[anonymization] model download failed: {e}")

    # 2.2 Load from disk
    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            use_fast=True,
            local_files_only=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            device_map="auto" if DEVICE=="cuda" else None,
            torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
        ).to(DEVICE)
        print("[anonymization] local LLM loaded successfully")
    except Exception as e:
        print(f"[anonymization] local LLM load failed: {e}")
        _tokenizer = None
        _model     = None


# ─── 3. Span detection via local LLM ────────────────────────────────────────────

def detect_sensitive_spans(text: str):
    """
    Uses the local LLM to find every span of PII/legal data.
    Returns a list of {"type":str, "start":int, "end":int}.
    """
    _ensure_local_llm()
    if not (_tokenizer and _model):
        return []

    prompt = (
        "You are a data-privacy assistant. Given the user’s text, identify every span of sensitive data:\n"
        "- emails, phone numbers, IP addresses, credit cards, SSNs or national IDs\n"
        "- person names, organizations, locations, dates\n"
        "- clause references (e.g. “Clause 5.3”)\n"
        "- case numbers (e.g. “Case No. 2021-CR-04567”)\n\n"
        "Return ONLY a JSON array of objects with keys:\n"
        "  type (string), start (int), end (int)\n\n"
        f"Text:\n'''{text}'''"
    )

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(DEVICE)
    outputs = _model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=_tokenizer.eos_token_id,
    )
    generated = _tokenizer.decode(outputs[0], skip_special_tokens=True)

    # extract JSON array
    try:
        js = generated[generated.index("["): generated.rindex("]")+1]
        spans = json.loads(js)
    except Exception:
        return []

    # validate spans
    valid = []
    for s in spans:
        if (
            isinstance(s.get("type"), str)
            and isinstance(s.get("start"), int)
            and isinstance(s.get("end"),   int)
            and 0 <= s["start"] < s["end"] <= len(text)
        ):
            valid.append(s)
    return valid


# ─── 4. The anonymize_text() API ────────────────────────────────────────────────

def anonymize_text(text: str):
    """
    1. Runs detect_sensitive_spans(text) via your local LLM.
    2. Replaces each span with <TYPE_n> (per-type counter).
    3. Returns (anonymized_text, mapping_list).
    """
    spans = detect_sensitive_spans(text)
    # reverse sort ensures earlier replacements don’t shift later indices
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

        out = out[:start] + token + out[end:]

    return out, mapping
