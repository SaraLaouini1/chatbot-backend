import re
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from transformers import pipeline
from collections import defaultdict

# 2 HF pipelines for legal NER
PIPELINES = [
    pipeline("ner", model="opennyaiorg/en_legal_ner_sm", aggregation_strategy="simple"),
    pipeline("ner", model="dslim/bert-base-NER-legal-contracts", aggregation_strategy="simple")
]

# Only these labels matter
LEGAL_LABELS = {
    'PARTY', 'CLAUSE', 'TERM', 'LAW', 'COURT',
    'CONTRACT', 'JUDGE', 'CASE_NUMBER', 'CLIENT_ID'
}

class LegalRecognizer:
    def analyze(self, text: str, entities=None, **kwargs):
        # Collect and dedupe HF spans
        spans = {}
        for pipe in PIPELINES:
            for ent in pipe(text):
                label = ent['entity_group'].upper()
                if label not in LEGAL_LABELS:
                    continue
                key = (ent['start'], ent['end'])
                if key not in spans or ent['score'] > spans[key]['score']:
                    spans[key] = {'label': label, **ent}
        # Build RecognizerResult list
        return [
            RecognizerResult(
                entity_type=span['label'],
                start=span['start'],
                end=span['end'],
                score=span['score']
            ) for span in spans.values()
        ]

# Remove overlaps
def filter_overlaps(ents):
    ordered = sorted(ents, key=lambda x: (x.start, -x.end))
    out, last_end = [], -1
    for e in ordered:
        if e.start >= last_end:
            out.append(e)
            last_end = e.end
        elif e.score > out[-1].score:
            out[-1] = e
            last_end = e.end
    return out

# Main anonymization
def anonymize_text(text: str):
    provider = NlpEngineProvider(nlp_engine=LegalRecognizer())
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
    # Clear default recognizers
    for r in list(analyzer.registry.recognizers):
        analyzer.registry.remove_recognizer(r)
    analyzer.registry.add_recognizer(LegalRecognizer())

    # Detect
    results = analyzer.analyze(text=text, language="en", score_threshold=0.85)
    filtered = filter_overlaps(results)
    filtered = sorted(filtered, key=lambda x: x.start, reverse=True)

    mapping, counts = {}, defaultdict(int)
    anonymized = text
    details = []

    for e in filtered:
        orig = text[e.start:e.end]
        key = (orig, e.entity_type)
        if key not in mapping:
            counts[e.entity_type] += 1
            tag = f"<{e.entity_type}_{counts[e.entity_type]}>"
            mapping[key] = tag
            details.append({"type": e.entity_type, "original": orig, "anonymized": tag})
        anonymized = anonymized[:e.start] + mapping[key] + anonymized[e.end:]

    return anonymized, details
