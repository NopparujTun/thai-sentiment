# Thai Sentiment and Intent Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/HuggingFace-F9AB00?logo=huggingface&logoColor=white" alt="Hugging Face"/>
  <img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" alt="Pandas"/>
</p>

An automated Natural Language Processing (NLP) pipeline designed to classify customer sentiment and categorize common business intents from Thai text. The system features custom fine-tuned transformer models, a fast inference pipeline, and a modern, interactive Streamlit analytics dashboard.

## Features

- **Sentiment Classification**: Predicts `Positive`, `Neutral`, and `Negative` sentiment using the Wisesight Sentiment Corpus.
- **Intent Classification**: Categorizes feedback into `Refund Request`, `Delivery Issue`, `Product Defect`, and `Product Question` using a custom annotated dataset.
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
│   └── intent.yaml
├── data/                   # Datasets
│   ├── raw/                # Raw datasets (git-ignored)
│   └── processed/          # Cleaned CSV files ready for training (git-ignored)
├── models/                 # Fine-tuned WangchanBERTa checkpoints (git-ignored)
│   ├── sentiment/
│   └── intent/
├── notebooks/              # Jupyter notebooks for data exploration
│   └── EDA.ipynb           # Exploratory Data Analysis
├── src/                    # Source code
│   ├── app/                # Streamlit UI
│   │   └── dashboard.py
│   ├── pipeline/           # Data prep and training
│   │   ├── augment_data.py
│   │   ├── prepare_data.py
│   │   ├── preprocessing.py
│   │   ├── train_intent.ipynb
│   │   └── train_sentiment.ipynb
│   └── predict/            # Inference scripts
│       ├── batch_predict.py
│       ├── predict_intent.py
│       └── predict_sentiment.py
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
To download the Wisesight corpus and generate the custom intent dataset:
```bash
PYTHONPATH=. python3 src/pipeline/prepare_data.py
```

### 2. Train the Sentiment Model
Fine-tunes WangchanBERTa for sentiment classification (Target F1: ≥ 85%).
```bash
jupyter notebook src/pipeline/train_sentiment.ipynb
```

### 3. Train the Intent Classifier
Fine-tunes WangchanBERTa for specific business intent categorization (Target F1: ≥ 80%).
```bash
jupyter notebook src/pipeline/train_intent.ipynb
```

## Inference

**Single Prediction:**
```bash
PYTHONPATH=. python3 src/predict/predict_sentiment.py "สินค้าดีมาก ชอบมาก"
PYTHONPATH=. python3 src/predict/predict_intent.py "ทำไมของยังไม่ถึง"
```

**Batch CSV Prediction:**
```bash
PYTHONPATH=. python3 src/predict/batch_predict.py data/processed/input.csv data/processed/output.csv
```

## Analytics Dashboard

Launch the interactive UI to upload CSV files of customer reviews and view live, interactive charts:
```bash
PYTHONPATH=. python3 -m streamlit run src/app/dashboard.py
```
> The dashboard will be hosted locally at `http://localhost:8501`.

## License
MIT License
