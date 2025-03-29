from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, EntityRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from transformers import pipeline
from collections import defaultdict
import re

# Initialize Presidio Analyzer
analyzer = AnalyzerEngine()

# Load BERT-based Named Entity Recognition (NER)
#bert_ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
bert_ner = pipeline("ner", model="dslim/bert-base-NER")


# Define a regex pattern for detecting passwords
PASSWORD_REGEX = r"(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}"

class BERTRecognizer(EntityRecognizer):
    def load(self):
        return

    def analyze(self, text, entities, language, nlp_artifacts=None):
        results = bert_ner(text)
        detected_entities = [
            {
                "entity_type": "PASSWORD" if res["entity"] == "MISC" else res["entity"],
                "start": res["start"],
                "end": res["end"],
                "score": res["score"],
            }
            for res in results if res["score"] > 0.85
        ]

        # Add regex-based password detection
        for match in re.finditer(PASSWORD_REGEX, text):
            detected_entities.append({
                "entity_type": "PASSWORD",
                "start": match.start(),
                "end": match.end(),
                "score": 0.95  # High confidence for regex matches
            })

        return detected_entities

def enhance_recognizers():
    """Enhance Presidio with BERT and regex-based recognition."""
    # Add BERT-based recognizer
    analyzer.registry.add_recognizer(BERTRecognizer(supported_entities=["PASSWORD", "ID", "EMAIL_ADDRESS"]))

def anonymize_text(text):
    enhance_recognizers()

    entities = ["PERSON", "PASSWORD", "EMAIL_ADDRESS", "CREDIT_CARD", "DATE_TIME", 
               "LOCATION", "PHONE_NUMBER", "NRP", "MONEY", "IBAN_CODE", "IP_ADDRESS", 
               "MEDICAL_LICENSE", "URL"]

    analysis = analyzer.analyze(
        text=text,
        entities=entities,
        language="en",
        score_threshold=0.3
    )

    analysis = sorted(analysis, key=lambda x: x.start, reverse=True)
    
    entity_counters = defaultdict(int)
    existing_mappings = {}
    anonymized_text = text

    for entity in analysis:
        entity_text = text[entity.start:entity.end]

        key = (entity_text, entity.entity_type)

        if key not in existing_mappings:
            entity_counters[entity.entity_type] += 1
            anonymized_label = f"<{entity.entity_type}_{entity_counters[entity.entity_type]}>"
            existing_mappings[key] = anonymized_label

        anonymized_text = (
            anonymized_text[:entity.start] + 
            existing_mappings[key] + 
            anonymized_text[entity.end:]
        )

    return anonymized_text, existing_mappings
