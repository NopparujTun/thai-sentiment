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

def main(config_path="configs/sentiment.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize W&B
    wandb.init(project=config["wandb_project"], name=config["wandb_run_name"])

    # Load Data
    print("Loading data...")
    df = pd.read_csv(config["data_path"])
    # Ensure no missing texts
    df = df.dropna(subset=['texts', 'category'])
    
    # Stratified split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['category'])
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    def tokenize_function(examples):
        return tokenizer(examples["texts"], padding="max_length", truncation=True, max_length=config["max_length"])

    print("Tokenizing data...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Rename and format for PyTorch
    train_dataset = train_dataset.rename_column("category", "labels")
    val_dataset = val_dataset.rename_column("category", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], 
        num_labels=3
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
