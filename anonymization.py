from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.transformers_recognizer import TransformersRecognizer
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from transformers import AutoModelForTokenClassification, AutoTokenizer
from collections import defaultdict
import spacy
import re

class LegalNlpEngine(SpacyNlpEngine):
    """Legal document processing engine with fallback"""
    def __init__(self):
        try:
            # Try loading large English model
            super().__init__(models={"en": "en_core_web_lg"})
        except OSError:
            print("Downloading base English model...")
            from spacy.cli import download
            download("en_core_web_lg")
            super().__init__(models={"en": "en_core_web_lg"})

# Initialize analyzer with legal configuration
analyzer = AnalyzerEngine(
    nlp_engine=LegalNlpEngine(),
    supported_languages=["en"]
)

def enhance_legal_recognizers():
    """Add legal-specific entity recognizers using Legal-BERT"""
    legal_bert = TransformersRecognizer(
        model_path="nlpaueb/legal-bert-base-uncased",
        aggregation_strategy="max",
        supported_entities=["PARTY", "CLAUSE_REF", "CONTRACT_TERM", "CASE_NUMBER"],
        context=["agreement", "section", "subsection", "witnesseth"]
    )
    
    # Configure confidence thresholds
    legal_bert.load_transformer(**{
        "model_to_confidence": {
            "PARTY": 0.92,
            "CLAUSE_REF": 0.88,
            "CONTRACT_TERM": 0.85,
            "CASE_NUMBER": 0.95
        }
    })
    
    analyzer.registry.add_recognizer(legal_bert)

def legal_context_validation(text, entity):
    """Validate entities using legal document context patterns"""
    context_rules = {
        "PARTY": [
            r"\bparty\b", r"\bbetween\b", r"\bsignatory\b", 
            r"\bhereinafter\b", r"\bwitnesseth\b"
        ],
        "CLAUSE_REF": [
            r"\bsection\b", r"\bclause\b", r"\barticle\b", 
            r"\bsubsection\b", r"\bparagraph\b"
        ],
        "CONTRACT_TERM": [
            r"\bterm\b", r"\beffective date\b", 
            r"\bexpiration\b", r"\brenewal\b"
        ],
        "CASE_NUMBER": [
            r"\bcase no\.?\b", r"\bdocket number\b", 
            r"\bfile ref\.?\b", r"\bindex no\.?\b"
        ]
    }
    
    context_window = text[max(0, entity.start-100):entity.end+100].lower()
    patterns = context_rules.get(entity.entity_type, [])
    
    return any(re.search(pattern, context_window) for pattern in patterns)

def anonymize_text(text):
    """Main anonymization function for legal documents"""
    enhance_legal_recognizers()
    
    # Analyze text with combined models
    entities = analyzer.analyze(
        text=text,
        language="en",
        score_threshold=0.8,
        return_decision_process=True
    )
    
    # Validate entities in legal context
    validated_entities = [
        ent for ent in entities
        if legal_context_validation(text, ent)
    ]
    
    # Anonymization processing
    replacements = {}
    entity_counter = defaultdict(int)
    anonymized = text
    
    # Process entities in reverse order to maintain positions
    for ent in sorted(validated_entities, key=lambda x: x.start, reverse=True):
        original = text[ent.start:ent.end]
        ent_type = ent.entity_type
        
        if original not in replacements:
            entity_counter[ent_type] += 1
            replacements[original] = f"<{ent_type}_{entity_counter[ent_type]}>"
        
        anonymized = (
            anonymized[:ent.start] + 
            replacements[original] + 
            anonymized[ent.end:]
        )
    
    # Prepare mapping documentation
    mapping = [
        {
            "type": ent_type,
            "original": original,
            "anonymized": replacement
        } 
        for original, replacement in replacements.items()
    ]
    
    return anonymized, mapping
