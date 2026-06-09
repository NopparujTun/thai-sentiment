# Project Overview

## Context
Thai Customer Feedback Analyzer is a Natural Language Processing (NLP) application designed to automatically analyze customer reviews and feedback in Thai. The system aims to automate the tedious manual review process by classifying both Customer Sentiment and Customer Issue Categories. It demonstrates a practical NLP workflow covering transformer fine-tuning, experiment tracking, reproducibility, and deployment via an interactive dashboard.

## Tech Stack
- **Programming Language**: Python 3.12+
- **NLP Frameworks**: PyTorch, Transformers, PyThaiNLP, SentencePiece
- **Foundation Model**: WangchanBERTa (used for both sentiment and issue classification)
- **Data Processing**: Pandas, NumPy
- **Visualization & UI**: Plotly, Streamlit
- **Experiment Tracking**: Weights & Biases (W&B)
- **Development Tools**: uv, Ruff, Black

## Architecture
The system follows a linear batch-processing pipeline:
`Raw Reviews` -> `Data Cleaning` -> `Tokenization` -> `WangchanBERTa` -> `[Sentiment Classifier | Issue Classifier]` -> `Prediction Results` -> `Analytics Dashboard`

---

# Procedure

## Phase 1 — Data Preparation

### Task 1.1: Collect and Clean Datasets
- **Objective**: Prepare the Wisesight Sentiment Corpus and create the custom Thai issue dataset.
- **Description**: Collect raw datasets, perform Exploratory Data Analysis (EDA), and clean the data according to the PRD (remove URLs, duplicate rows, empty rows, and normalize whitespace).
- **Implementation**:
  - *Simplicity First*: Write minimal, reproducible scripts for URL and whitespace removal using Pandas and regex. Avoid speculative data augmentation features not requested.
  - *Goal-Driven Execution*:
    1. Implement cleaning functions -> verify: unit tests pass for `remove_urls` and `normalize_whitespace`.
    2. Process raw datasets -> verify: processed datasets have no missing values, URLs, or duplicates, and are saved to `data/processed/`.
- **Deliverables**: Processed, clean datasets ready for training.

---

## Phase 2 — Sentiment Model

### Task 2.1: Fine-tune Sentiment Classifier
- **Objective**: Train a sentiment classifier using WangchanBERTa on the Wisesight dataset.
- **Description**: Configure the model, run the training pipeline for Positive, Neutral, and Negative classes, track metrics, and evaluate.
- **Implementation**:
  - *Surgical Changes*: Limit code changes strictly to the model configuration and training script. Avoid building unnecessary abstractions around the Hugging Face Trainer.
  - *Goal-Driven Execution*:
    1. Define config in `configs/sentiment.yaml` -> verify: configuration parses correctly.
    2. Run `train_sentiment.py` -> verify: training executes without errors, and W&B logs Accuracy, Precision, Recall, and F1.
    3. Evaluate model -> verify: F1 Score reaches ≥ 85%.
- **Deliverables**: Fine-tuned sentiment classifier model checkpoint, W&B logs, and evaluation report.

---

## Phase 3 — Issue Model

### Task 3.1: Annotate and Fine-tune Issue Classifier
- **Objective**: Develop an issue classifier using the custom dataset.
- **Description**: Annotate 300-500 samples per class based on label priority (Refund Request > Delivery Issue > Product Defect > Product Question). Fine-tune WangchanBERTa and evaluate.
- **Implementation**:
  - *Think Before Coding*: Assume annotation guidelines are strictly followed. State explicit prioritization during the annotation phase to prevent label ambiguity.
  - *Goal-Driven Execution*:
    1. Annotate dataset -> verify: class distribution meets the target of 300-500 samples per class.
    2. Run `train_issue.py` -> verify: model trains successfully and logs correctly to W&B.
    3. Evaluate model -> verify: Macro F1 Score reaches ≥ 80%.
- **Deliverables**: Fine-tuned issue classifier model checkpoint, and the custom annotated Thai issue dataset.

---

## Phase 4 — Inference Pipeline

### Task 4.1: Develop Single and Batch Prediction
- **Objective**: Build the inference pipeline to generate predictions from text or CSV files.
- **Description**: Create inference scripts that load the trained checkpoints and output predictions along with confidence scores.
- **Implementation**:
  - *Simplicity First*: Write a straightforward inference function. Do not overcomplicate with streaming or real-time queues, as real-time streaming is explicitly excluded in the PRD.
  - *Goal-Driven Execution*:
    1. Implement single prediction -> verify: passing sample text returns correct schema (Label, Confidence Score).
    2. Implement batch prediction -> verify: processing a CSV exports a new CSV containing `sentiment`, `sentiment_score`, `issue`, and `issue_score` under 2 seconds per single prediction equivalent.
- **Deliverables**: Prediction pipeline scripts (`predict_sentiment.py`, `predict_issue.py`, `batch_predict.py`) and CSV export functionality.

---

## Phase 5 — Dashboard

### Task 5.1: Build Analytics Dashboard
- **Objective**: Create a Streamlit dashboard to visualize inference results and upload datasets.
- **Description**: Implement a UI supporting CSV uploads, displaying total reviews, sample predictions, and interactive Plotly charts (Pie, Bar, Frequency).
- **Implementation**:
  - *Think Before Coding*: Design the dashboard to handle up to 5,000 reviews smoothly. Focus purely on functionality defined in FR-06.
  - *Goal-Driven Execution*:
    1. Implement CSV upload component -> verify: file is parsed and the `review_text` column is validated.
    2. Integrate batch inference -> verify: dashboard correctly runs prediction on uploaded data.
    3. Implement Plotly visualizations -> verify: charts render accurately based on prediction distributions.
- **Deliverables**: An operational, interactive Streamlit analytics dashboard.
