# anonymization.py
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from collections import defaultdict
import re
from langdetect import detect
import spacy

# Initialize NLP engine with both French and English models
def setup_nlp_engine():
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "fr", "model_name": "fr_core_news_md"},
                {"lang_code": "en", "model_name": "en_core_web_lg"}
            ]
        }
    )
    return provider.create_engine()

nlp_engine = setup_nlp_engine()
analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=["fr", "en"]
)

CURRENCY_NORMALIZATION = {
    "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "usd": "USD", "dollars": "USD", "dollar": "USD",
    "gbp": "GBP", "livre": "GBP", "livres": "GBP",
    "mad": "MAD", "dh": "MAD", "dirham": "MAD", "dirhams": "MAD"
}

def enhance_recognizers():
    # Money recognizer with bilingual context
    money_recognizer = PatternRecognizer(
        supported_entity="MONEY",
        patterns=[
            Pattern(
                "money_pattern",
                r"(?i)(\d+)\s*(\$|€|£|USD|EUR|GBP|MAD)|\b(\d+)\s?(dollars|euros|livres|dirhams|dh)\b",
                0.9
            )
        ],
        context=["invoice", "amount", "payment", "facture", "montant", "paiement"]
    )

    # Phone number recognizer for international formats
    phone_recognizer = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[
            Pattern(
                "international_phone",
                r"(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}",
                0.9
            )
        ],
        context=["phone", "mobile", "tel", "number", "téléphone", "portable"]
    )

    analyzer.registry.add_recognizer(money_recognizer)
    analyzer.registry.add_recognizer(phone_recognizer)

def detect_language(text):
    try:
        lang = detect(text)  
        return lang if lang in ['fr', 'en'] else 'en'
    except:
        return 'en'

def anonymize_text(text):
    lang = detect_language(text)
    enhance_recognizers()
    
    entities = [
        "PERSON", "EMAIL_ADDRESS", "CREDIT_CARD", "DATE_TIME",
        "LOCATION", "PHONE_NUMBER", "MONEY"
    ]

    analysis = analyzer.analyze(
        text=text,
        entities=entities,
        language=lang,
        score_threshold=0.4
    )

    analysis = sorted(analysis, key=lambda x: x.start, reverse=True)
    
    entity_counters = defaultdict(int)
    updated_analysis = []
    existing_mappings = {}
    anonymized_text = text

    for entity in analysis:
        entity_text = text[entity.start:entity.end]

        # Normalization for money values
        if entity.entity_type == "MONEY":
            entity_text = re.sub(
                r"(\d+)\s*([a-zA-Z]+)",
                lambda m: f"{m.group(1)} {CURRENCY_NORMALIZATION.get(m.group(2).lower(), m.group(2).upper()}",
                entity_text
            )

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

    return anonymized_text, updated_analysis, lang
