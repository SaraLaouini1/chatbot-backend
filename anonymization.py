# anonymization.py

from presidio_analyzer import AnalyzerEngine, RecognizerResult, EntityRecognizer
from collections import defaultdict
import re

# 1️⃣ Define your custom NER-recognizer for Legal-BERT, but do *not* load the model yet
class LegalBertRecognizer(EntityRecognizer):
    def __init__(self):
        super().__init__(
            supported_entities=["PARTY", "CLAUSE_REF", "CONTRACT_TERM", "CASE_NUMBER"],
            name="legal-bert-recognizer",
            supported_language="en"
        )
        self.ner = None   # ← model will be created on first use

    def _ensure_model(self):
        if self.ner is None:
            from transformers import pipeline
            try:
                self.ner = pipeline(
                    "token-classification",
                    model="nlpaueb/legal-bert-base-uncased",
                    aggregation_strategy="simple",
                    device=-1                # force CPU (no GPU) to avoid OOM on small servers
                )
            except Exception as e:
                # if model load fails, just skip it
                print(f"[LegalBertRecognizer] failed to load model: {e}")
                self.ner = []

    def analyze(self, text, entities, *, nlp_artifacts=None, language=None):
        # lazy-load
        self._ensure_model()
        if not self.ner:
            return []

        results = []
        for ent in self.ner(text):
            label = {
                "PER": "PARTY", "ORG": "PARTY",
                "DATE": "CONTRACT_TERM", "MONEY": "CONTRACT_TERM",
                "LOC": "CLAUSE_REF"
            }.get(ent["entity_group"])
            if not label or label not in entities:
                continue
            results.append(RecognizerResult(
                entity_type=label,
                start=ent["start"],
                end=ent["end"],
                score=ent["score"],
            ))
        return results

# 2️⃣ Initialize the engine *without* triggering any model-loads
analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(LegalBertRecognizer())

def legal_context_validation(text: str, ent) -> bool:
    rules = {
        "PARTY":        [r"\bparty\b", r"\bbetween\b", r"\bherein\b"],
        "CLAUSE_REF":   [r"\bsection\b", r"\bclause\b"],
        "CONTRACT_TERM":[r"\bterm\b", r"\beffective date\b"],
        "CASE_NUMBER":  [r"\bcase no\.?\b", r"\bdocket\b"]
    }
    window = text[max(0, ent.start - 50): ent.end + 50].lower()
    return any(re.search(p, window) for p in rules.get(ent.entity_type, []))

def anonymize_text(text: str):
    # 1. Detect *all* built-in PII + your Legal-BERT spans
    entities = analyzer.analyze(
        text=text,
        language="en",
        score_threshold=0.8
    )
    # 2. Filter out any non-legal contexts if you still want
    entities = [e for e in entities if legal_context_validation(text, e)]
    # 3. Build mappings and redact
    existing = {}
    counters = defaultdict(int)
    updated = []
    out = text
    for e in sorted(entities, key=lambda x: x.start, reverse=True):
        orig = out[e.start:e.end]
        key = (orig, e.entity_type)
        if key not in existing:
            counters[e.entity_type] += 1
            tok = f"<{e.entity_type}_{counters[e.entity_type]}>"
            existing[key] = tok
            updated.append({"type":e.entity_type, "original":orig, "anonymized":tok})
        else:
            tok = existing[key]
        out = out[:e.start] + tok + out[e.end:]
    return out, updated
