from presidio_analyzer import AnalyzerEngine, TransformersRecognizer
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from transformers import AutoModelForTokenClassification, AutoTokenizer
from collections import defaultdict
import spacy
import re

class LegalNlpEngine(SpacyNlpEngine):
    """English legal document processing engine"""
    def __init__(self):
        # Load or download legal model
        try:
            super().__init__(models={"en": "en_legal_core_ml_md"})
        except OSError:
            print("Downloading legal model...")
            from spacy.cli import download
            download("en_legal_core_ml_md")
            super().__init__(models={"en": "en_legal_core_ml_md"})

# Initialize analyzer with legal configuration
analyzer = AnalyzerEngine(
    nlp_engine=LegalNlpEngine(),
    supported_languages=["en"]
)

def enhance_legal_recognizers():
    """Add legal-specific entity recognizers"""
    # Legal BERT model for contract analysis
    legal_bert_recognizer = TransformersRecognizer(
        model_path="nlpaueb/legal-bert-small-uncased",
        aggregation_strategy="max",
        supported_entities=["PARTY", "CLAUSE_REF", "CONTRACT_TERM"]
    )
    
    # Configure confidence thresholds
    legal_bert_recognizer.load_transformer(**{
        "model_to_confidence": {
            "PARTY": 0.95,
            "CLAUSE_REF": 0.92,
            "CONTRACT_TERM": 0.88
        }
    })
    
    analyzer.registry.add_recognizer(legal_bert_recognizer)

def legal_context_validation(text, entity):
    """Validate entities using legal document context"""
    context_rules = {
        "PARTY": ["party", "hereinafter", "between", "witnesseth"],
        "CLAUSE_REF": ["section", "clause", "article", "subsection"],
        "CONTRACT_TERM": ["term", "effective date", "expiration", "renewal"]
    }
    
    context_window = text[max(0, entity.start-150):entity.end+150].lower()
    return any(keyword in context_window 
              for keyword in context_rules.get(entity.entity_type, []))

def anonymize_text(text):
    """Main anonymization function for legal documents"""
    enhance_legal_recognizers()
    
    # Analyze with higher threshold for legal precision
    entities = analyzer.analyze(
        text=text,
        language="en",
        score_threshold=0.85,
        return_decision_process=True
    )
    
    # Contextual validation
    validated_entities = [ent for ent in entities 
                        if legal_context_validation(text, ent)]
    
    # Anonymization logic
    entity_counter = defaultdict(int)
    replacements = {}
    anonymized = text
    
    for ent in sorted(validated_entities, key=lambda x: x.start, reverse=True):
        ent_type = ent.entity_type
        original = text[ent.start:ent.end]
        
        if (original, ent_type) not in replacements:
            entity_counter[ent_type] += 1
            replacements[(original, ent_type)] = f"<{ent_type}_{entity_counter[ent_type]}>"
        
        anonymized = (
            anonymized[:ent.start] + 
            replacements[(original, ent_type)] + 
            anonymized[ent.end:]
        )
    
    # Prepare mapping documentation
    mapping = [
        {
            "type": ent_type,
            "original": original,
            "anonymized": replacement
        } for (original, ent_type), replacement in replacements.items()
    ]
    
    return anonymized, mapping
