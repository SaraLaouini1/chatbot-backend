# anonymization.py

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import TransformersRecognizer
from collections import defaultdict
import re

# 1️⃣ Initialize Presidio with its default spaCy engine.
analyzer = AnalyzerEngine()

def enhance_legal_recognizers():
    """
    Register a single TransformersRecognizer on top of the spaCy engine.
    Legal‑BERT will pick up parties, clause refs, contract dates/terms, and case numbers.
    """
    legal_bert = TransformersRecognizer(
        model_name="nlpaueb/legal-bert-base-uncased",
        tokenizer_name="nlpaueb/legal-bert-base-uncased",
        aggregation_strategy="max",
        supported_entities=[
            "PARTY",        # e.g. “Acme Corp”
            "CLAUSE_REF",   # e.g. “Section 5.1”
            "CONTRACT_TERM",# e.g. “January 1, 2025”
            "CASE_NUMBER"   # e.g. “2023‑ABC‑123”
        ],
        threshold=0.85
    )
    analyzer.registry.add_recognizer(legal_bert)

def legal_context_validation(text: str, ent) -> bool:
    """
    Quick regex‑based guard: only keep spans that appear with legal keywords nearby.
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
    1. Ensure the Legal‑BERT recognizer is loaded
    2. Run the analysis
    3. (Optional) Filter by legal context
    4. Replace each span with <TYPE_n> in reverse order
    5. Return anonymized text + mapping list
    """
    enhance_legal_recognizers()

    # ▶️ 2. Detect
    entities = analyzer.analyze(
        text=text,
        entities=["PARTY", "CLAUSE_REF", "CONTRACT_TERM", "CASE_NUMBER"],
        language="en",
        score_threshold=0.8,
    )

    # ▶️ 3. Context filter (keep only those that live in legal context)
    entities = [e for e in entities if legal_context_validation(text, e)]

    # ▶️ 4. Replace spans back→front
    replacements = {}
    counters = defaultdict(int)
    result = text

    for e in sorted(entities, key=lambda x: x.start, reverse=True):
        span = result[e.start:e.end]
        typ = e.entity_type
        if (span, typ) not in replacements:
            counters[typ] += 1
            replacements[(span, typ)] = f"<{typ}_{counters[typ]}>"
        token = replacements[(span, typ)]
        result = result[:e.start] + token + result[e.end:]

    # ▶️ 5. Prepare mapping
    mapping = [
        {"type": typ, "original": orig, "anonymized": token}
        for (orig, typ), token in replacements.items()
    ]

    return result, mapping
