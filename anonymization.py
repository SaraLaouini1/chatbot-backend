from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider  # Changed import
from collections import defaultdict
import re

# Initialize spaCy engine CORRECTLY
configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
}

provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()

analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

# Dictionary to standardize currency names
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

# Custom recognizers
def enhance_recognizers():
    # Money format recognizer
    money_pattern = Pattern(
        name="money_pattern",
        regex=r"(?i)(\d+)\s*(\$|€|£|USD|EUR|GBP|MAD)|\b(\d+)\s?(dollars|euros|pounds|dirhams|dh)\b",
        score=0.9
    )
    money_recognizer = PatternRecognizer(
        supported_entity="MONEY",
        patterns=[money_pattern],
        context=["invoice", "amount", "payment"]
    )

    # Custom Credit Card Recognizer (without Luhn check)
    credit_card_pattern = Pattern(
        name="credit_card_pattern",
        regex=r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",
        score=0.9
    )
    credit_card_recognizer = PatternRecognizer(
        supported_entity="CREDIT_CARD",
        patterns=[credit_card_pattern],
        context=["card", "credit", "account"]
    )

    # Add new recognizer for spaCy entities
    org_recognizer = PatternRecognizer(
        supported_entity="ORG",
        context=["company", "organization", "firm"],  # Context words boost detection
        deny_list=[]  # Add your custom organizations
    )


    # Add to enhance_recognizers()
    password_pattern = Pattern(
        name="password_pattern",
        regex=r"(?i)(password|passwd|pwd)[\s:]+([^\s]{8,})|\b[a-f0-9]{32}\b",
        score=0.85
    )
    password_recognizer = PatternRecognizer(
        supported_entity="PASSWORD",
        patterns=[password_pattern],
        context=["login", "credentials"]
    )

    analyzer.registry.add_recognizer(password_recognizer)
    analyzer.registry.add_recognizer(org_recognizer)
    analyzer.registry.add_recognizer(credit_card_recognizer)
    analyzer.registry.add_recognizer(money_recognizer)

def normalize_money_format(money_str):
    """Normalize different currency representations to avoid duplicates."""
    match = re.search(r"(\d+)\s*([a-zA-Z]+)", money_str)
    if match:
        amount, currency = match.groups()
        normalized_currency = CURRENCY_NORMALIZATION.get(currency.lower(), currency.upper())
        return f"{amount} {normalized_currency}"
    return money_str

def anonymize_text(text):
    enhance_recognizers()
    
    entities = ["PERSON","PASSWORD", "EMAIL_ADDRESS", "CREDIT_CARD", "DATE_TIME", 
               "LOCATION", "PHONE_NUMBER", "NRP", "MONEY", "URL", "IBAN_CODE", "IP_ADDRESS",
               
                "ORG", "NORP", "LANGUAGE", "EVENT", "LAW"
               ]

    analysis = analyzer.analyze(
        text=text,
        entities=entities,
        language="en",
        score_threshold=0.45,  # Better balance than 0.35
        return_decision_process=False  # Disable for production
    )

    # Sort entities in reverse order to prevent replacement conflicts
    analysis = sorted(analysis, key=lambda x: x.start, reverse=True)
    
    entity_counters = defaultdict(int)
    updated_analysis = []
    existing_mappings = {}
    anonymized_text = text

    for entity in analysis:
        entity_text = text[entity.start:entity.end]
        
        # Normalize money values
        if entity.entity_type == "MONEY":
            entity_text = normalize_money_format(entity_text)

        # Create unique key with entity type and text
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

        # Replace in text
        anonymized_text = (
            anonymized_text[:entity.start] + 
            existing_mappings[key] + 
            anonymized_text[entity.end:]
        )

    return anonymized_text, updated_analysis
