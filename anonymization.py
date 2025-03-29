# Install required packages
# pip install presidio-analyzer transformers torch datasets

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.transformers_recognizer import TransformersRecognizer
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
from collections import defaultdict
import random
import string

# ==================================================================
# 1. Custom Model Training (Run this once to create detection models)
# ==================================================================

def generate_training_data(num_samples=500):
    """Generate synthetic training data for passwords and IDs"""
    data = []
    for _ in range(num_samples):
        # Generate random context patterns
        contexts = [
            f"Password: {''.join(random.choices(string.ascii_letters + string.digits + '!@#$%^&*', k=12))}",
            f"User ID: {''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}",
            f"Security token: {random.randint(1000,9999)}-{''.join(random.choices(string.ascii_letters, k=6))}",
            f"Access code: {''.join(random.choices(string.digits, k=10))}"
        ]
        
        for text in contexts:
            entities = []
            if "Password" in text:
                start = text.find(":") + 2
                end = len(text)
                entities.append((start, end, "PASSWORD"))
            elif "ID" in text:
                start = text.find(":") + 2
                end = len(text)
                entities.append((start, end, "ID"))
            
            data.append((text, {"entities": entities}))
    
    return Dataset.from_dict({
        "text": [d[0] for d in data],
        "entities": [d[1]["entities"] for d in data]
    })

# Initialize base model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=3,  # 0: O, 1: B-PASSWORD, 2: B-ID
    id2label={0: "O", 1: "B-PASSWORD", 2: "B-ID"},
    label2id={"O": 0, "B-PASSWORD": 1, "B-ID": 2}
)

# Convert training data to tokenized format
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    labels = []
    
    for i, entity_list in enumerate(examples["entities"]):
        text = examples["text"][i]
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [0] * len(word_ids)
        
        for start, end, label in entity_list:
            token_start = None
            token_end = None
            for idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if start <= tokenizer.decode(tokenized_inputs["input_ids"][i][idx], skip_special_tokens=True).start():
                    token_start = idx
                    break
            for idx, word_id in enumerate(reversed(word_ids)):
                if word_id is None:
                    continue
                if end >= len(text) - tokenizer.decode(tokenized_inputs["input_ids"][i][::-1][idx], skip_special_tokens=True).start():
                    token_end = len(word_ids) - idx
                    break
            
            if token_start and token_end:
                label_type = 1 if label == "PASSWORD" else 2
                for j in range(token_start, token_end):
                    label_ids[j] = label_type
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Train the model
dataset = generate_training_data().map(tokenize_and_align_labels, batched=True)
training_args = TrainingArguments(
    output_dir="./security_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained("./security_model")
tokenizer.save_pretrained("./security_model")

# ==============================================
# 2. Anonymization Code with ML Detection
# ==============================================

class SecurityNlpEngine(TransformersNlpEngine):
    def __init__(self):
        super().__init__(
            models=[
                {"model_name": "dslim/bert-base-NER", "labels": ["LOC", "MISC", "ORG", "PER"]},
                {"model_name": "./security_model", "labels": ["PASSWORD", "ID"]}
            ]
        )

analyzer = AnalyzerEngine(nlp_engine=SecurityNlpEngine())

# Add custom recognizers for our ML-detected entities
security_recognizer = TransformersRecognizer(
    model_path="./security_model",
    supported_entities=["PASSWORD", "ID"],
    context=["credentials", "authentication", "access", "security"]
)

analyzer.registry.add_recognizer(security_recognizer)

def anonymize_text(text):
    entities = ["PASSWORD", "ID", "PER", "LOC", "ORG"]
    
    analysis = analyzer.analyze(
        text=text,
        entities=entities,
        language="en",
        score_threshold=0.85
    )

    # Sort entities in reverse order to prevent replacement conflicts
    analysis = sorted(analysis, key=lambda x: x.start, reverse=True)
    
    entity_counters = defaultdict(int)
    updated_analysis = []
    existing_mappings = {}
    anonymized_text = text

    for entity in analysis:
        entity_text = text[entity.start:entity.end]
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

