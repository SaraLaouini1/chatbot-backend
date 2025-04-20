# anonymization.py

from presidio_analyzer import AnalyzerEngine
# ↪️ import TransformersRecognizer from the top‐level predefined_recognizers
from presidio_analyzer.predefined_recognizers import TransformersRecognizer

from collections import defaultdict
import re

# 1️⃣ Initialize Presidio with its default spaCy engine
analyzer = AnalyzerEngine()

def enhance_legal_recognizers():
    """
    Register the Legal‑BERT based TransformersRecognizer.
    """
    recognizer = TransformersRecognizer(
        # this __init__ only accepts model_path & supported_entities
        model_path="nlpaueb/legal-bert-base-uncased",
        supported_entities=[
            "PARTY",        # e.g. “Acme Corp”
            "CLAUSE_REF",   # e.g. “Section 5.1”
            "CONTRACT_TERM",# e.g. “January 1, 2025”
            "CASE_NUMBER"   # e.g. “2023‑ABC‑123”
        ],
    )
    # now configure the HF pipeline
    recognizer.load_transformer(
        SUB_WORD_AGGREGATION="simple",
        CHUNK_SIZE=600,
        CHUNK_OVERLAP_SIZE=40,
        LABELS_TO_IGNORE=["O"],
    )
    analyzer.registry.add_recognizer(recognizer)

def legal_context_validation(text: str, ent) -> bool:
    """
    Keep only those spans that appear near typical legal keywords.
    """
    rules = {
        "PARTY":        [r"\bparty\b", r"\bbetween\b", r"\bherein\b"],
        "CLAUSE_REF":   [r"\bsection\b", r"\bclause\b"],
        "CONTRACT_TERM":[r"\bterm\b", r"\beffective date\b"],
        "CASE_NUMBER":  [r"\bcase no\.?\b", r"\bdocket\b"]
    }
    window = text[max(0, ent.start - 50): ent.end + 50].lower()
    return any(re.search(p, window) for p in rules.get(ent.entity_type, []))

def anonymize_text(text: str):
    """
    1. Register the Legal‑BERT recognizer
    2. Run Presidio analysis
    3. (Optional) Filter by legal context
    4. Splice out each span (back→front) with <TYPE_n>
    5. Return (anonymized_text, mapping)
    """
    enhance_legal_recognizers()

    # 2️⃣ Detect
    entities = analyzer.analyze(
        text=text,
        entities=["PARTY", "CLAUSE_REF", "CONTRACT_TERM", "CASE_NUMBER"],
        language="en",
        score_threshold=0.8,
    )

    # 3️⃣ Filter false positives via regex context
    entities = [e for e in entities if legal_context_validation(text, e)]

    # 4️⃣ Replace spans back→front
    replacements = {}
    counters = defaultdict(int)
    result = text

    for e in sorted(entities, key=lambda x: x.start, reverse=True):
        span = result[e.start:e.end]
        typ  = e.entity_type
        if (span, typ) not in replacements:
            counters[typ] += 1
            replacements[(span, typ)] = f"<{typ}_{counters[typ]}>"
        token = replacements[(span, typ)]
        result = result[:e.start] + token + result[e.end:]

    # 5️⃣ Build mapping for re‑injection
    mapping = [
        {"type": typ, "original": orig, "anonymized": token}
        for (orig, typ), token in replacements.items()
    ]

    return result, mapping
