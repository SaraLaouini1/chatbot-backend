# anonymization.py

from presidio_analyzer import AnalyzerEngine
# 👉 correct import path for version 2.2.x
from presidio_analyzer.predefined_recognizers.transformers_recognizer import TransformersRecognizer

from collections import defaultdict
import re

# 1️⃣ Initialize Presidio with its default spaCy engine
analyzer = AnalyzerEngine()

def enhance_legal_recognizers():
    """
    Add a single TransformersRecognizer (Legal‑BERT) to the registry.
    """
    recognizer = TransformersRecognizer(
        model_path="nlpaueb/legal-bert-base-uncased",
        # only these two args are valid in __init__:
        supported_entities=[
            "PARTY",        # e.g. “Acme Corp”
            "CLAUSE_REF",   # e.g. “Section 5.1”
            "CONTRACT_TERM",# e.g. “January 1, 2025”
            "CASE_NUMBER"   # e.g. “2023‑ABC‑123”
        ],
    )
    # configure the HF pipeline (you can also add MODEL_TO_PRESIDIO_MAPPING here if needed)
    recognizer.load_transformer(**{
        "SUB_WORD_AGGREGATION": "simple",        # how to merge subword tokens
        "CHUNK_SIZE": 600,                       # max chars per inference chunk
        "CHUNK_OVERLAP_SIZE": 40,                # overlap between chunks
        "LABELS_TO_IGNORE": ["O"],               # ignore the 'O' label
        # optional: you can set:
        # "DATASET_TO_PRESIDIO_MAPPING": {...},
        # "MODEL_TO_PRESIDIO_MAPPING": {...},
        # "ID_ENTITY_NAME": "CASE_NUMBER", etc.
    })
    analyzer.registry.add_recognizer(recognizer)

def legal_context_validation(text: str, ent) -> bool:
    """
    Only keep spans if they occur near legal keywords.
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
    1. Register legal recognizer
    2. Run analysis
    3. (Optional) Filter by context
    4. Replace spans back→front
    5. Return (anonymized_text, mapping)
    """
    enhance_legal_recognizers()

    # ▶️ 2. Detect all four entity types
    entities = analyzer.analyze(
        text=text,
        entities=["PARTY", "CLAUSE_REF", "CONTRACT_TERM", "CASE_NUMBER"],
        language="en",
        score_threshold=0.8,
    )

    # ▶️ 3. Filter false positives via simple regex context
    entities = [e for e in entities if legal_context_validation(text, e)]

    # ▶️ 4. Anonymize from back→front
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

    # ▶️ 5. Build mapping list
    mapping = [
        {"type": typ, "original": orig, "anonymized": token}
        for (orig, typ), token in replacements.items()
    ]

    return result, mapping
