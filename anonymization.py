# anonymization.py
import spacy
from transformers import pipeline
import hashlib
from collections import defaultdict

# Legal-specific entity types
LEGAL_ENTITY_TYPES = {
    "PARTY_NAME", "CASE_NUMBER", "JURISDICTION", 
    "CLAUSE_REFERENCE", "EFFECTIVE_DATE", "LEGAL_CITATION",
    "FINANCIAL_TERM", "CONTRACT_VALUE", "IDENTIFICATION_NUMBER"
}

# Load local legal NLP models
LEGAL_NLP = spacy.load("en_legal_core_web_md")
NER_PIPELINE = pipeline("ner", model="dslim/bert-large-NER", aggregation_strategy="average")

class LegalAnonymizer:
    def __init__(self):
        self.entity_map = defaultdict(lambda: defaultdict(str))
        self.legal_context_terms = {
            'party', 'clause', 'hereinafter', 'witnesseth',
            'exhibit', 'whereas', 'notwithstanding', 'agreement'
        }

    def _detect_legal_entities(self, text: str) -> list:
        """Hybrid legal entity detection with type classification"""
        entities = []
        
        # spaCy legal model detection
        doc = LEGAL_NLP(text)
        for ent in doc.ents:
            entity_type = self._classify_legal_entity(ent.text, ent.label_, text)
            if entity_type:
                entities.append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "type": entity_type
                })
        
        # Transformers NER detection
        transformer_results = NER_PIPEline(text)
        for res in transformer_results:
            if res['score'] > 0.85:
                entity_type = self._classify_legal_entity(res['word'], res['entity_group'], text)
                if entity_type:
                    entities.append({
                        "text": res['word'],
                        "start": res['start'],
                        "end": res['end'],
                        "type": entity_type
                    })
        
        return entities

    def _classify_legal_entity(self, text: str, label: str, context: str) -> str:
        """Map detected entities to legal-specific types"""
        context = context.lower()
        
        # Legal entity type mapping
        if label in ["PERSON", "PER"]:
            if any(term in context for term in ["party", "client", "signatory"]):
                return "PARTY_NAME"
            
        elif label in ["ORG", "LEGAL_ORG"]:
            if "party" in context:
                return "PARTY_NAME"
            return "LEGAL_ENTITY"
            
        elif label == "DATE":
            if any(term in context for term in ["effective", "termination", "commencement"]):
                return "EFFECTIVE_DATE"
                
        elif label == "LAW":
            return "LEGAL_CITATION"
            
        elif label == "CARDINAL":
            if "clause" in context:
                return "CLAUSE_REFERENCE"
            if "case" in context:
                return "CASE_NUMBER"
                
        elif label == "MONEY":
            return "FINANCIAL_TERM"
            
        return None

    def _generate_typed_pseudonym(self, text: str, entity_type: str) -> str:
        """Create type-specific pseudonym with consistent hashing"""
        salt = hashlib.sha256(text.encode()).hexdigest()[:8]
        return f"<{entity_type}_{salt}>"

    def anonymize(self, text: str) -> tuple:
        """Type-preserving legal anonymization"""
        entities = self._detect_legal_entities(text)
        mapping = []
        text_chars = list(text)
        
        # Process entities from longest to shortest
        for entity in sorted(entities, key=lambda x: x['end']-x['start'], reverse=True):
            original = entity['text']
            entity_type = entity['type']
            
            if not self.entity_map[entity_type].get(original):
                pseudonym = self._generate_typed_pseudonym(original, entity_type)
                self.entity_map[entity_type][original] = pseudonym
                mapping.append({
                    "original": original,
                    "anonymized": pseudonym,
                    "type": entity_type
                })
            
            start = entity['start']
            end = entity['end']
            text_chars[start:end] = list(self.entity_map[entity_type][original])
        
        return "".join(text_chars), mapping
