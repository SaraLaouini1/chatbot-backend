# anonymization.py

from presidio_analyzer import AnalyzerEngine, RecognizerResult, EntityRecognizer
from transformers import pipeline
from collections import defaultdict
import re

# 1️⃣ Define a tiny custom NER‑recognizer for Legal‑BERT
class LegalBertRecognizer(EntityRecognizer):
    def __init__(self):
        super().__init__(
            supported_entities=["PARTY", "CLAUSE_REF", "CONTRACT_TERM", "CASE_NUMBER"],
            name="legal-bert-recognizer",
            supported_language="en"
        )
        self.ner = pipeline(
            "token-classification",
            model="nlpaueb/legal-bert-base-uncased",
            aggregation_strategy="simple"
        )
        self.label_map = {
            "PER":           "PARTY",
            "ORG":           "PARTY",
            "DATE":          "CONTRACT_TERM",
            "MONEY":         "CONTRACT_TERM",
            "LOC":           "CLAUSE_REF"
            # Extend here for CASE_NUMBER, if applicable
        }


    def analyze(self, text, entities, *, nlp_artifacts=None, language=None):
        results = []
        for ent in self.ner(text):
            label = self.label_map.get(ent["entity_group"])
            if not label or label not in entities:
                continue
            start, end, score = ent["start"], ent["end"], ent["score"]
            results.append(RecognizerResult(
                entity_type=label,
                start=start,
                end=end,
                score=score
            ))
        return results


# 2️⃣ Initialize the engine and register your custom recognizer
analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(LegalBertRecognizer())


def legal_context_validation(text: str, ent) -> bool:
    """Optional: keep only those spans near legal keywords."""
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
    1. Run Presidio analysis (with LegalBertRecognizer active)
    2. (Optional) Filter by legal_context_validation
    3. Replace each span back→front with <TYPE_n>, tracking only *new* mappings
    4. Return anonymized text + updated_analysis (new mappings)
    """
    # ▶️ 1. Detect
    entities = analyzer.analyze(
        text=text,
        entities=["PARTY", "CLAUSE_REF", "CONTRACT_TERM", "CASE_NUMBER"],
        language="en",
        score_threshold=0.8
    )

    # ▶️ 2. (Optional) Context filter
    entities = [e for e in entities if legal_context_validation(text, e)]

    # ▶️ 3. Prepare mapping structures
    existing_mappings = {}             # key: (orig, type) → anonymized_label
    entity_counters   = defaultdict(int)
    updated_analysis  = []             # only new mappings go here

    # work on a mutable copy
    anonymized_text = text

    # ▶️ 4. Splice out spans back→front (reverse offset so indices remain valid)
    for ent in sorted(entities, key=lambda x: x.start, reverse=True):
        orig        = anonymized_text[ent.start:ent.end]
        key         = (orig, ent.entity_type)

        if key not in existing_mappings:
            # first time we see this exact text+type
            entity_counters[ent.entity_type] += 1
            anonymized_label = f"<{ent.entity_type}_{entity_counters[ent.entity_type]}>"
            existing_mappings[key] = anonymized_label

            # record it in updated_analysis
            updated_analysis.append({
                "type":       ent.entity_type,
                "original":   orig,
                "anonymized": anonymized_label
            })
        else:
            anonymized_label = existing_mappings[key]

        # do the actual replacement
        anonymized_text = anonymized_text[:ent.start] + anonymized_label + anonymized_text[ent.end:]

    # ▶️ 5. Return both the final text and only the *new* mappings we created
    return anonymized_text, updated_analysis
