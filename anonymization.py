from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, EntityRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from collections import defaultdict
import re
from transformers import pipeline

# Configure NLP engine with spaCy
provider = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]
})
nlp_engine = provider.create_engine()

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=["en"]
)

CURRENCY_NORMALIZATION = {
    "eur": "EUR",
    "euro": "EUR",
    "usd": "USD",
    "dollars": "USD",
    "dh": "MAD",
    "dirham": "MAD",
    "gbp": "GBP",
    "pounds": "GBP"
}

class TransformersRecognizer(EntityRecognizer):
    """ML-powered recognizer using Hugging Face transformers"""
    def __init__(self):
        super().__init__(supported_entities=["PROFESSIONAL_STATUS", "INTERNAL_ID"], name="HF Transformers")
        self.model = pipeline("token-classification", model="dslim/bert-base-NER")
        
    def load(self):
        pass
    
    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        predictions = self.model(text)
        for pred in predictions:
            if pred['entity_group'] in self.supported_entities:
                results.append({
                    'start': pred['start'],
                    'end': pred['end'],
                    'score': pred['score'],
                    'entity_type': pred['entity_group']
                })
        return results

def enhance_recognizers():
    """Register all custom pattern recognizers"""
    # Enhanced phone number recognizer
    phone_recognizer = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[
            Pattern(
                "international_phone",
                r"\+(?:[0-9]●?){6,14}[0-9]",  # Matches international format
                0.9
            ),
            Pattern(
                "standard_phone",
                r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",  # US/CA numbers
                0.85
            )
        ],
        context=["phone", "mobile", "tel", "number", "contact"]
    )

    # Professional status recognizer
    professional_status_recognizer = PatternRecognizer(
        supported_entity="PROFESSIONAL_STATUS",
        patterns=[Pattern("status_pattern", r"\b(Full-time|Part-time|Contract|Freelance|Consultant)\b", 0.7)],
        context=["employment", "position", "role", "status"]
    )

    # General financial recognizers
    credit_card_recognizer = PatternRecognizer(
        supported_entity="CREDIT_CARD",
        patterns=[Pattern("cc_pattern", r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b", 0.95)],
        context=["card", "credit", "account", "payment"]
    )

    recognizers = [
        phone_recognizer,
        professional_status_recognizer,
        credit_card_recognizer,
        TransformersRecognizer()
    ]
    
    for recognizer in recognizers:
        analyzer.registry.add_recognizer(recognizer)

def normalize_money_format(money_str):
    """Normalize currency representations"""
    match = re.search(r"(\d+)\s*([a-zA-Z]+)", money_str)
    if match:
        amount, currency = match.groups()
        normalized_currency = CURRENCY_NORMALIZATION.get(currency.lower(), currency.upper())
        return f"{amount} {normalized_currency}"
    return money_str

def anonymize_text(text):
    enhance_recognizers()
    
    entities = [
        "PERSON", "EMAIL_ADDRESS", "CREDIT_CARD", "DATE_TIME",
        "LOCATION", "PHONE_NUMBER", "MONEY", "PROFESSIONAL_STATUS",
        "INTERNAL_ID"
    ]

    analysis = analyzer.analyze(
        text=text,
        entities=entities,
        language="en",
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
