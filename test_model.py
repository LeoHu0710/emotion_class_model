import os
import json
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from log import train_and_test_logger

mapping_table = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

## loading model and tokenizer
checkpoint_path = os.path.join("results", "checkpoint_results")
train_and_test_logger.info(f"model path : {checkpoint_path}")

model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)

model_name = "roberta-base"
train_and_test_logger.info(f"model : {model_name}")
tokenizer = RobertaTokenizer.from_pretrained(model_name)

## load test data
test_file = os.path.join("emotion", "test.jsonl")
texts, labels = [], []

with open(test_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        texts.append(data["text"])
        labels.append(data["label"])

## predict
predictions = []
with torch.no_grad():
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        logits = model(**inputs).logits
        predictions.append(torch.argmax(logits, dim=1).item())

## score
accuracy = accuracy_score(labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

train_and_test_logger.info(f"Accuracy: {accuracy:.4f}")
train_and_test_logger.info(f"Precision: {precision:.4f}")
train_and_test_logger.info(f"Recall: {recall:.4f}")
train_and_test_logger.info(f"F1-score: {f1:.4f}")

## confusion matrix
cm = confusion_matrix(labels, predictions)
print("\nConfusion Matrix:")
print(cm)

train_and_test_logger.info("\nConfusion Matrix:")
train_and_test_logger.info(cm)

