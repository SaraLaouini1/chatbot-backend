# anonymization.py
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from typing import List, Optional, Tuple, Dict
import spacy
from transformers import pipeline
from collections import defaultdict
import re

# Legal domain configuration
LEGAL_ENTITY_TYPES = {
    'CASE_NUMBER', 'CLIENT_ID', 'CONTRACT_ID', 
    'LEGAL_ENTITY', 'JUDICIAL_OFFICER', 'COURT_CODE',
    'CONFIDENTIAL_CLAUSE', 'LEGAL_REFERENCE'
}

class LegalNlpEngine:
    """Custom NLP engine for legal document analysis"""
    def __init__(self):
        self.spacy_model = spacy.load("en_legal_ner_trf")
        self.contract_ner = pipeline(
            "ner",
            model="dslim/bert-base-NER-legal-contracts",
            aggregation_strategy="max"
        )
        self.entity_mapping = {
            'PARTY': 'LEGAL_ENTITY',
            'JUDGE': 'JUDICIAL_OFFICER',
            'CLAUSE': 'CONFIDENTIAL_CLAUSE',
            'COURT': 'COURT_CODE'
        }

    def analyze(self, text: str) -> List[Dict]:
        """Hybrid legal entity recognition"""
        # SpaCy analysis
        spacy_doc = self.spacy_model(text)
        spacy_ents = [{
            "text": ent.text,
            "label": self.entity_mapping.get(ent.label_, ent.label_),
            "start": ent.start_char,
            "end": ent.end_char,
            "score": 0.95
        } for ent in spacy_doc.ents]

        # Transformers analysis
        hf_ents = self.contract_ner(text)
        hf_ents = [{
            "text": e["word"],
            "label": self.entity_mapping.get(e["entity_group"], e["entity_group"]),
            "start": e["start"],
            "end": e["end"],
            "score": e["score"]
        } for e in hf_ents]

        return self._merge_entities(spacy_ents + hf_ents)

    def _merge_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge and deduplicate entities from different models"""
        merged = []
        sorted_ents = sorted(entities, key=lambda x: (x["start"], -x["end"]))
        
        for ent in sorted_ents:
            if not merged or ent["start"] >= merged[-1]["end"]:
                if ent["label"] in LEGAL_ENTITY_TYPES:
                    merged.append(ent)
            else:
                current = merged[-1]
                if ent["score"] > current["score"] and ent["label"] in LEGAL_ENTITY_TYPES:
                    merged[-1] = ent
        return merged

class LegalRecognizer:
    """Presidio recognizer adapter for legal entities"""
    def analyze(self, text: str, entities: List[str]) -> List[RecognizerResult]:
        legal_engine = LegalNlpEngine()
        entities = legal_engine.analyze(text)
        
        return [
            RecognizerResult(
                entity_type=ent["label"],
                start=ent["start"],
                end=ent["end"],
                score=ent["score"]
            ) for ent in entities
        ]

def filter_overlapping_entities(entities: List[RecognizerResult]) -> List[RecognizerResult]:
    """Remove overlapping entities keeping highest confidence"""
    entities = sorted(entities, key=lambda x: (x.start, -x.end))
    filtered = []
    last_end = -1
    
    for ent in entities:
        if ent.start >= last_end:
            filtered.append(ent)
            last_end = ent.end
        else:
            if ent.score > filtered[-1].score:
                filtered[-1] = ent
                last_end = ent.end
    return filtered

def anonymize_text(text: str) -> Tuple[str, List[Dict]]:
    """Complete legal document anonymization pipeline"""
    # Configure NLP engine with legal models
    provider = NlpEngineProvider(nlp_engine=LegalNlpEngine())
    analyzer = AnalyzerEngine(
        nlp_engine=provider.create_engine(),
        supported_languages=["en"]
    )
    analyzer.registry.add_recognizer(LegalRecognizer())

    # Detect sensitive entities
    entities = analyzer.analyze(
        text=text,
        language="en",
        entities=list(LEGAL_ENTITY_TYPES),
        score_threshold=0.85
    )
    
    # Process entities
    entities = filter_overlapping_entities(entities)
    entities = sorted(entities, key=lambda x: x.start, reverse=True)
    
    # Anonymization mapping
    mapping = defaultdict(int)
    anonymized = text
    audit_trail = []

    for ent in entities:
        entity_text = text[ent.start:ent.end]
        
        # Generate consistent placeholder
        mapping_key = f"{ent.entity_type}_{mapping[ent.entity_type]}"
        placeholder = f"<{mapping_key}>"
        mapping[ent.entity_type] += 1
        
        # Replace text
        anonymized = (
            anonymized[:ent.start] + 
            placeholder + 
            anonymized[ent.end:]
        )
        
        # Record audit trail
        audit_trail.append({
            "type": ent.entity_type,
            "original": entity_text,
            "anonymized": placeholder,
            "confidence": ent.score,
            "start": ent.start,
            "end": ent.end
        })
    
    return anonymized, audit_trail

def reidentify_text(anonymized: str, audit_trail: List[Dict]) -> str:
    """Reconstruct original text from anonymized version"""
    restored = anonymized
    for item in reversed(audit_trail):
        restored = restored.replace(
            item["anonymized"],
            item["original"]
        )
    return restored
