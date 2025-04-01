from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from collections import defaultdict
import re

analyzer = AnalyzerEngine()

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

def remove_default_money_recognizer():
    # Gather all recognizers that detect MONEY
    money_recognizers = [r for r in analyzer.registry.recognizers 
                         if "MONEY" in r.supported_entities]
    # Remove each one
    for rec in money_recognizers:
        analyzer.registry.remove_recognizer(rec)


def filter_overlapping_entities(entities):
    # Sort entities by starting index, then by longer span
    entities = sorted(entities, key=lambda x: (x.start, -x.end))
    filtered = []
    last_end = -1
    for ent in entities:
        if ent.start >= last_end:
            filtered.append(ent)
            last_end = ent.end
        else:
            # Overlap detected; optionally choose the entity with a higher score.
            if ent.score > filtered[-1].score:
                filtered[-1] = ent
                last_end = ent.end
    return filtered


# Custom recognizers
def enhance_recognizers():
    remove_default_money_recognizer()  # Remove any built-in money recognizers
    
    # Money format recognizer
    money_pattern = Pattern(
        name="money_pattern",
        regex=r"(?i)(?:(?<=\s)|(?<=^))(?:(?:\$|€|£|USD|EUR|GBP|MAD)\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:\$|€|£|USD|EUR|GBP|MAD)|\d+(?:\.\d+)?\s*(?:dollars|euros|pounds|dirhams|dh))\b",
        score=0.99
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
    
    entities = analyzer.analyze(
        text=text,
        entities=["PERSON", "EMAIL_ADDRESS", "CREDIT_CARD", "DATE_TIME", 
                  "LOCATION", "PHONE_NUMBER", "NRP", "MONEY", "URL"],
        language="en",
        score_threshold=0.3
    )
    
    # Filter out overlapping entities
    entities = filter_overlapping_entities(entities)
    
    # Sort entities in reverse order to avoid index shifts
    entities = sorted(entities, key=lambda x: x.start, reverse=True)
    
    entity_counters = defaultdict(int)
    updated_analysis = []
    existing_mappings = {}
    anonymized_text = text

    for entity in entities:
        entity_text = text[entity.start:entity.end]
        
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
        
        # Replace entity in the text using slicing in reverse order
        anonymized_text = (
            anonymized_text[:entity.start] +
            existing_mappings[key] +
            anonymized_text[entity.end:]
        )
    
    return anonymized_text, updated_analysis

