from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from collections import defaultdict
import re

analyzer = AnalyzerEngine()

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

def remove_default_money_recognizer():
    default_recognizers = analyzer.registry.recognizers
    for rec in default_recognizers:
        if "MONEY" in rec.supported_entities:
            analyzer.registry.remove_recognizer(rec)

def filter_overlapping_entities(entities):
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

def enhance_recognizers():
    remove_default_money_recognizer()
    
    # Simplified money pattern with strict boundaries
    money_pattern = Pattern(
        name="money_pattern",
        regex=r"(?i)(?<!#)(?:€|\$|£|USD|EUR|GBP|MAD)\s*\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:dollars|euros|pounds|dirhams|dh)\b(?!\d)",
        score=0.95
    )


    money_recognizer = PatternRecognizer(
        supported_entity="MONEY",
        patterns=[money_pattern],
        context=["invoice", "amount", "payment"]
    )

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

    analyzer.registry.add_recognizer(credit_card_recognizer)
    analyzer.registry.add_recognizer(money_recognizer)

def normalize_money_format(money_str):
    match = re.search(r"(\d+)\s*([a-zA-Z]+)", money_str)
    if match:
        amount, currency = match.groups()
        normalized_currency = CURRENCY_NORMALIZATION.get(currency.lower(), currency.upper())
        return f"{amount} {normalized_currency}"
    return money_str

def anonymize_text(text):
    enhance_recognizers()
    
    entities = analyzer.analyze(
        text=text,
        entities=["PERSON", "EMAIL_ADDRESS", "CREDIT_CARD", "DATE_TIME",
                  "LOCATION", "PHONE_NUMBER", "NRP", "MONEY", "URL", "IP_ADDRESS"],
        language="en",
        score_threshold=0.3
    )
    
    entities = filter_overlapping_entities(entities)
    entities = sorted(entities, key=lambda x: x.start, reverse=True)
    
    entity_counters = defaultdict(int)
    updated_analysis = []
    existing_mappings = {}
    anonymized_text = text

    for entity in entities:
        entity_text = text[entity.start:entity.end]
        
        if entity.entity_type == "MONEY":
            if not re.search(r"(?:€|\$|£|USD|EUR|GBP|MAD|dollars|euros|pounds|dirhams|dh)", entity_text, re.I):
                continue
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
