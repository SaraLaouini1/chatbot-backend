# anonymization.py (Updated)
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from typing import List, Optional
import spacy
import torch
from transformers import pipeline

# Load legal NLP models
LEGAL_NER_MODEL = "joelito/legal-ner"
CONTRACT_NER_MODEL = "dslim/bert-base-NER-legal-contracts"

class LegalNlpEngine:
    def __init__(self):
        self.spacy_model = spacy.load("en_legal_ner_trf")
        self.hf_model = pipeline(
            "ner", 
            model=CONTRACT_NER_MODEL,
            aggregation_strategy="max"
        )
        self.legal_entities = {
            'PARTY', 'CLAUSE', 'TERM', 'LAW', 'COURT', 
            'CONTRACT', 'JUDGE', 'CASE_NUMBER', 'CLIENT_ID'
        }

    def analyze_legal_text(self, text: str) -> List[dict]:
        """Analyze text using multiple legal NLP models"""
        # SpaCy analysis
        spacy_doc = self.spacy_model(text)
        spacy_ents = [{
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "score": 0.95
        } for ent in spacy_doc.ents]

        # Transformers analysis
        hf_ents = self.hf_model(text)
        merged_ents = self._merge_results(spacy_ents, hf_ents)
        
        return [
            ent for ent in merged_ents
            if ent["label"] in self.legal_entities
            and self._validate_context(text, ent)
        ]

    def _merge_results(self, spacy_ents, hf_ents):
        # Advanced merging logic for model results
        merged = []
        for ent in hf_ents:
            ent["label"] = ent["entity_group"]
            merged.append(ent)
        for ent in spacy_ents:
            if not any(self._overlap(ent, e) for e in merged):
                merged.append(ent)
        return merged

    def _overlap(self, ent1, ent2):
        return not (ent1["end"] <= ent2["start"] or ent2["end"] <= ent1["start"])

    def _validate_context(self, text: str, entity: dict) -> bool:
        """Validate entity using contextual analysis"""
        context_window = text[max(0, entity["start"]-50):entity["end"]+50]
        context_keywords = {
            "CASE_NUMBER": ["case", "docket", "number", "v."],
            "CLIENT_ID": ["client", "id", "confidential", "matter"],
            "CONTRACT": ["agreement", "party", "clause", "section"]
        }
        
        keywords = context_keywords.get(entity["label"], [])
        return any(kw in context_window.lower() for kw in keywords)

class LegalRecognizer:
    def analyze(self, text: str, entities: List[str]) -> List[RecognizerResult]:
        legal_engine = LegalNlpEngine()
        entities = legal_engine.analyze_legal_text(text)
        
        return [
            RecognizerResult(
                entity_type=ent["label"],
                start=ent["start"],
                end=ent["end"],
                score=ent["score"]
            ) for ent in entities
        ]

def anonymize_text(text: str):
    # Initialize analyzer with legal recognizers
    provider = NlpEngineProvider(nlp_engine=LegalNlpEngine())
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
    analyzer.registry.add_recognizer(LegalRecognizer())
    
    # Analyze with combined legal and default entities
    entities = analyzer.analyze(
        text=text,
        language="en",
        score_threshold=0.85,
        return_decision_process=True
    )
    
    # Anonymization logic remains similar with additional legal entities
    # ... (rest of your existing anonymization logic)
