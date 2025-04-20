from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import TransformersRecognizer
from collections import defaultdict
import re

# 1️⃣ Init with Presidio’s default spaCy engine (it'll load en_core_web_sm by default).
#    If you want en_core_web_lg, just pre-download it in your build.
analyzer = AnalyzerEngine()

def enhance_legal_recognizers():
    """Add just one TransformersRecognizer for Legal‑BERT—no custom spaCy engine needed."""
    legal_bert = TransformersRecognizer(
        model_name="nlpaueb/legal-bert-base-uncased",    # HF model
        tokenizer_name="nlpaueb/legal-bert-base-uncased",
        aggregation_strategy="max",                      # pick highest‐score token span
        supported_entities=[
            "PARTY",       # e.g. “Acme Corp”
            "CLAUSE_REF",  # e.g. “Section 5.1”
            "CONTRACT_TERM", 
            "CASE_NUMBER"
        ],
        threshold=0.85                                    # tune for high precision
    )
    analyzer.registry.add_recognizer(legal_bert)

def legal_context_validation(text, ent):
    """Optional: ensure the found span really lives in a legal context."""
    rules = {
        "PARTY":        [r"\bparty\b", r"\bbetween\b", r"\bherein\b"],
        "CLAUSE_REF":   [r"\bsection\b", r"\bclause\b"],
        "CONTRACT_TERM":[r"\bterm\b", r"\beffective date\b"],
        "CASE_NUMBER":  [r"\bcase no\.?\b", r"\bdocket\b"]
    }
    window = text[max(0, ent.start-50):ent.end+50].lower()
    return any(re.search(p, window) for p in rules.get(ent.entity_type, []))

def anonymize_text(text: str):
    # 2️⃣ Ensure our recognizer is registered
    enhance_legal_recognizers()

    # 3️⃣ Run Presidio analysis
    entities = analyzer.analyze(
        text=text,
        entities=["PARTY", "CLAUSE_REF", "CONTRACT_TERM", "CASE_NUMBER"],
        language="en",
        score_threshold=0.8,
    )

    # 4️⃣ (Optional) filter by context
    entities = [e for e in entities if legal_context_validation(text, e)]

    # 5️⃣ Replace spans with <TYPE_n>
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

    # Build the mapping list
    mapping = [
        {"type": typ, "original": orig, "anonymized": token}
        for (orig, typ), token in replacements.items()
    ]

    return result, mapping
