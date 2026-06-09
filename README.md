# Thai Customer Feedback Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/HuggingFace-F9AB00?logo=huggingface&logoColor=white" alt="Hugging Face"/>
  <img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" alt="Pandas"/>
</p>

An automated Natural Language Processing (NLP) pipeline designed to classify customer sentiment and categorize common business issues from Thai text. The system features custom fine-tuned transformer models, a fast inference pipeline, and a modern, interactive Streamlit analytics dashboard.

## Features

- **Sentiment Classification**: Predicts `Positive`, `Neutral`, and `Negative` sentiment using the Wisesight Sentiment Corpus.
- **Issue Classification**: Categorizes feedback into `Refund Request`, `Delivery Issue`, `Product Defect`, and `Product Question` using a custom annotated dataset.
- **Fast Inference Pipeline**: Single predictions and scalable CSV batch processing.
- **Premium Analytics Dashboard**: A glassmorphism-styled Streamlit UI with Plotly charts for interactive data visualization.

## Tech Stack

- **Language**: Python 3.12+
- **Foundation Model**: [WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased) (PyThaiNLP/VISTEC)
- **NLP & Deep Learning**: PyTorch, Transformers, Datasets
- **Data Manipulation**: Pandas, Scikit-Learn
- **Dashboard & Visualization**: Streamlit, Plotly Express
- **Experiment Tracking**: Weights & Biases (W&B)

## Project Structure

```text
thai-sentiment/
├── configs/                # Hyperparameter configurations for training (YAML)
│   ├── sentiment.yaml
│   └── issue.yaml
├── data/                   # Datasets
│   ├── raw/                # Raw datasets (git-ignored)
│   └── processed/          # Cleaned CSV files ready for training (git-ignored)
├── models/                 # Fine-tuned WangchanBERTa checkpoints (git-ignored)
│   ├── sentiment/
│   └── issue/
├── src/                    # Source code
│   ├── prepare_data.py     # Data generation & pipeline execution
│   ├── preprocessing.py    # Text cleaning rules (URLs, whitespace)
│   ├── train_sentiment.py  # Sentiment model training script
│   ├── train_issue.py      # Issue model training script
│   ├── predict_sentiment.py# Single inference for sentiment
│   ├── predict_issue.py    # Single inference for issues
│   ├── batch_predict.py    # Batch CSV inference
│   └── dashboard.py        # Streamlit analytics dashboard UI
├── tests/                  # Unit tests (pytest)
├── plan.md                 # Original project execution plan
└── PRD.md                  # Product Requirements Document
```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NopparujTun/thai-sentiment.git
   cd thai-sentiment
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install pandas scikit-learn transformers datasets wandb accelerate pythainlp sentencepiece torch streamlit plotly
   ```

3. **Login to Weights & Biases (for training logs):**
   ```bash
   wandb login
   ```

## Model Training

### 1. Data Preparation
To download the Wisesight corpus and generate the custom issue dataset:
```bash
PYTHONPATH=. python3 src/prepare_data.py
```

### 2. Train the Sentiment Model
Fine-tunes WangchanBERTa for sentiment classification (Target F1: ≥ 85%).
```bash
PYTHONPATH=. python3 src/train_sentiment.py
```

### 3. Train the Issue Classifier
Fine-tunes WangchanBERTa for specific business issue categorization (Target F1: ≥ 80%).
```bash
PYTHONPATH=. python3 src/train_issue.py
```

## Inference

**Single Prediction:**
```bash
PYTHONPATH=. python3 src/predict_sentiment.py "สินค้าดีมาก ชอบมาก"
PYTHONPATH=. python3 src/predict_issue.py "ทำไมของยังไม่ถึง"
```

**Batch CSV Prediction:**
```bash
PYTHONPATH=. python3 src/batch_predict.py data/processed/input.csv data/processed/output.csv
```

## Analytics Dashboard

Launch the interactive UI to upload CSV files of customer reviews and view live, interactive charts:
```bash
PYTHONPATH=. python3 -m streamlit run src/dashboard.py
```
> The dashboard will be hosted locally at `http://localhost:8501`.

## License
MIT License
