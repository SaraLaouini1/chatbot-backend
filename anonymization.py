import re
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from transformers import pipeline
from huggingface_hub import snapshot_download
import spacy
from collections import defaultdict

# HF models for legal NER
token_models = {
    "legal": "joelito/legal-ner",
    "contract": "dslim/bert-base-NER-legal-contracts"
}

class LegalNlpEngine:
    def __init__(self):
        # 1️⃣ Load spaCy legal model from HF snapshot
        try:
            self.spacy_model = spacy.load("en_legal_ner_trf")
        except OSError:
            model_dir = snapshot_download("opennyaiorg/en_legal_ner_trf")
            self.spacy_model = spacy.load(model_dir)

        # 2️⃣ Load HF pipelines
        self.pipes = {
            name: pipeline("ner", model=path, aggregation_strategy="simple")
            for name, path in token_models.items()
        }
        # Keep only these entity labels
        self.legal_entities = {
            'PARTY', 'CLAUSE', 'TERM', 'LAW', 'COURT',
            'CONTRACT', 'JUDGE', 'CASE_NUMBER', 'CLIENT_ID'
        }

    def analyze_legal_text(self, text: str):
        # spaCy detections
        doc = self.spacy_model(text)
        sp_ents = [{
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "score": 0.95
        } for ent in doc.ents]

        # HF detections
        hf_ents = []
        for pipe in self.pipes.values():
            hf = pipe(text)
            for e in hf:
                e["label"] = e.pop("entity_group").upper()
            hf_ents.extend(hf)

        # Merge, dedupe by span
        merged = { (e['start'], e['end']): e for e in sp_ents }
        for e in hf_ents:
            key = (e['start'], e['end'])
            if key not in merged or e['score'] > merged[key]['score']:
                merged[key] = e

        # Filter to legal_labels
        return [e for e in merged.values() if e['label'] in self.legal_entities]

class LegalRecognizer:
    def analyze(self, text: str, entities=None, **kwargs):
        engine = LegalNlpEngine()
        results = engine.analyze_legal_text(text)
        return [ RecognizerResult(
                    entity_type=e['label'],
                    start=e['start'],
                    end=e['end'],
                    score=e['score']
                 ) for e in results ]

# Overlap filter
def filter_overlapping_entities(ents):
    ents = sorted(ents, key=lambda x: (x.start, -x.end))
    out, last_end = [], -1
    for e in ents:
        if e.start >= last_end:
            out.append(e)
            last_end = e.end
        elif e.score > out[-1].score:
            out[-1] = e
            last_end = e.end
    return out

# Main anonymization function
def anonymize_text(text: str):
    provider = NlpEngineProvider(nlp_engine=LegalNlpEngine())
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
    # strip defaults & add only legal recognizer
    for r in list(analyzer.registry.recognizers):
        analyzer.registry.remove_recognizer(r)
    analyzer.registry.add_recognizer(LegalRecognizer())

    # detect
    detected = analyzer.analyze(text=text, language="en", score_threshold=0.85)
    detected = filter_overlapping_entities(detected)
    # replace in reverse order
    detected = sorted(detected, key=lambda x: x.start, reverse=True)

    mapping, counters, anonymized = {}, defaultdict(int), text
    details = []
    for e in detected:
        orig = text[e.start:e.end]
        key = (orig, e.entity_type)
        if key not in mapping:
            counters[e.entity_type] += 1
            tag = f"<{e.entity_type}_{counters[e.entity_type]}>"
            mapping[key] = tag
            details.append({"type":e.entity_type, "original":orig, "anonymized":tag})
        anonymized = anonymized[:e.start] + mapping[key] + anonymized[e.end:]

    return anonymized, details
