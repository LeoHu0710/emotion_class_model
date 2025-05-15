from transformers import Trainer, TrainingArguments, RobertaTokenizer, RobertaForSequenceClassification
from datasets import Dataset
import os
import json
from log import train_and_test_logger

os.makedirs("results", exist_ok=True)

def load_data(jsonl_path):
    texts = []
    labels = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
            labels.append(data["label"])
    return texts, labels

train_path = os.path.join("emotion", "train.jsonl")
validation_path = os.path.join("emotion", "validation.jsonl")

## load train data and val data
try :
    train_texts, train_labels = load_data(train_path)
    validation_texts, validation_labels = load_data(validation_path)
    train_and_test_logger.info(f"Load data success. train data : {len(train_texts)}, validation data : {len(validation_texts)}")

except Exception as e :
    train_and_test_logger.error(f"Load data error : {e}")

## loading model and tokenizer
model_name = "roberta-base"
train_and_test_logger.info(f"model : {model_name}")
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels = 6)

## tokenizer
try :
    train_encodings = tokenizer(train_texts, padding="max_length", truncation=True)
    validation_encodings = tokenizer(validation_texts, padding="max_length", truncation=True)
    train_and_test_logger.info("Tokenizer success.")

except Exception as e :
    train_and_test_logger.error(f"Tokenizer error : {e}")

# Combine into a Dataset
try :
    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels
    })

    validation_dataset = Dataset.from_dict({
        "input_ids": validation_encodings["input_ids"],
        "attention_mask": validation_encodings["attention_mask"],
        "labels": validation_labels
    })
    train_and_test_logger.info("Table to create a [Dataset] -> success.")

except Exception as e :
    train_and_test_logger.error(f"Table to create a [Dataset] -> error :\n{e}")

## Set training parameters
training_args = TrainingArguments(
    output_dir = os.path.join(".", "results"),
    eval_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs=5,
    weight_decay = 0.01,
)
train_and_test_logger.info(f"training args :\n {training_args}")

## training start.
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = validation_dataset
)

train_and_test_logger.info(f"training start.")
trainer.train()
train_and_test_logger.info(f"training end.")