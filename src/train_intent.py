import pandas as pd
import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import wandb
import sys

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main(config_path="configs/intent.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize W&B
    wandb.init(project=config["wandb_project"], name=config["wandb_run_name"])

    print("Loading data...")
    df = pd.read_csv(config["data_path"])
    df = df.dropna(subset=['text', 'label'])
    
    # Label mapping according to priority
    label_map = {
        "Refund Request": 0,
        "Delivery Issue": 1,
        "Product Defect": 2,
        "Product Question": 3
    }
    df['label'] = df['label'].map(label_map)
    
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=config["max_length"])

    print("Tokenizing data...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], 
        num_labels=4
    )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"] * 2,
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        learning_rate=config["learning_rate"],
        eval_strategy=config["evaluation_strategy"],
        save_strategy=config["save_strategy"],
        logging_steps=config["logging_steps"],
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    
    print("Evaluating...")
    trainer.evaluate()
    
    print(f"Saving model to {config['output_dir']}...")
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
