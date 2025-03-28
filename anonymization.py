from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, EntityRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from collections import defaultdict
import re
from transformers import pipeline
from langdetect import detect

# Configure NLP engine with both English and French models
provider = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_lg"},
        {"lang_code": "fr", "model_name": "fr_core_news_md"}
    ]
})
nlp_engine = provider.create_engine()

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=["en", "fr"]
)

CURRENCY_NORMALIZATION = {
    "eur": "EUR", "euro": "EUR", "usd": "USD", "dollars": "USD",
    "gbp": "GBP", "pounds": "GBP"
}

class TransformersRecognizer(EntityRecognizer):
    """Multilingual ML-powered recognizer with entity group fix"""
    def __init__(self):
        super().__init__(
            supported_entities=["PER", "ORG", "LOC"],  # Actual entities from the model
            name="HF Transformers"
        )
        self.model = pipeline(
            "token-classification",
            model="Davlan/bert-base-multilingual-cased-ner-hrl"
        )
        
    def load(self):
        pass
    
    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        predictions = self.model(text)
        for pred in predictions:
            # Handle different model output formats
            entity_type = pred.get('entity_group', pred.get('entity'))
            
            if entity_type in self.supported_entities:
                results.append({
                    'start': pred['start'],
                    'end': pred['end'],
                    'score': pred['score'],
                    'entity_type': entity_type
                })
        return results

def enhance_recognizers():
    """Register multilingual recognizers"""
    # International phone number recognizer
    phone_recognizer = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[
            Pattern(
                "international_phone",
                r"(?:\+\d{1,3}[- ]?)?\d{2,4}[- ]?\d{3,4}[- ]?\d{3,4}",
                0.9
            )
        ],
        context=["phone", "mobile", "tel", "number", "contact",
                 "téléphone", "portable", "numéro"]
    )

    # Multilingual professional status recognizer (regex-based)
    professional_status_recognizer = PatternRecognizer(
        supported_entity="PROFESSIONAL_STATUS",
        patterns=[
            Pattern("status_en", r"\b(Full-time|Part-time|Contract|Freelance)\b", 0.7),
            Pattern("status_fr", r"\b(CDI|CDD|Stage|Indépendant)\b", 0.7)
        ],
        context=["employment", "position", "role", "status",
                 "emploi", "poste", "statut"]
    )

    # International credit card recognizer
    credit_card_recognizer = PatternRecognizer(
        supported_entity="CREDIT_CARD",
        patterns=[Pattern(
            "cc_pattern", 
            r"\b(?:\d[ -]*?){13,19}\b",  # Matches various credit card formats
            0.95
        )],
        context=["card", "credit", "account", "payment", 
                 "carte", "crédit", "paiement"]
    )

    recognizers = [
        phone_recognizer,
        professional_status_recognizer,
        credit_card_recognizer,
        TransformersRecognizer()
    ]
    
    for recognizer in recognizers:
        analyzer.registry.add_recognizer(recognizer)

def detect_language(text):
    """Detect text language with fallback to English"""
    try:
        lang = detect(text)
        return lang if lang in ['en', 'fr'] else 'en'
    except:
        return 'en'

def anonymize_text(text):
    enhance_recognizers()
    lang = detect_language(text)
    
    entities = [
        "PERSON", "EMAIL_ADDRESS", "CREDIT_CARD", "DATE_TIME",
        "LOCATION", "PHONE_NUMBER", "MONEY", "PROFESSIONAL_STATUS",
        "PER", "ORG"  # Added model entities
    ]

    analysis = analyzer.analyze(
        text=text,
        entities=entities,
        language=lang,
        score_threshold=0.45
    )

    analysis = sorted(analysis, key=lambda x: x.start, reverse=True)
    
    entity_counters = defaultdict(int)
    updated_analysis = []
    existing_mappings = {}
    anonymized_text = text

    for entity in analysis:
        entity_text = text[entity.start:entity.end]

        # Normalize monetary values
        if entity.entity_type == "MONEY":
            entity_text = normalize_money_format(entity_text)

        key = (entity_text, entity.entity_type)
        
        if key not in existing_mappings:
            entity_counters[entity.entity_type] += 1
            anonymized_label = f"<{entity.entity_type}_{entity_counters[entity.entity_type]}>"
            existing_mappings[key] = anonymized_label
            updated_analysis.append({
                "type": entity.entity_type,
                "original": entity_text,
                "anonymized": anonymized_label
            })

        anonymized_text = (
            anonymized_text[:entity.start] + 
            existing_mappings[key] + 
            anonymized_text[entity.end:]
        )

    return anonymized_text, updated_analysis

def normalize_money_format(money_str):
    """Normalize currency representations"""
    match = re.search(r"(\d+)\s*([a-zA-Z]+)", money_str)
    if match:
        amount, currency = match.groups()
        normalized_currency = CURRENCY_NORMALIZATION.get(currency.lower(), currency.upper())
        return f"{amount} {normalized_currency}"
    return money_str
