# Thai Customer Feedback Analyzer

## Product Requirements Document (PRD)

Version: 2.0 (Karpathy Guidelines Applied)

---

# 1. Project Overview & Assumptions (Think Before Coding)

**Don't assume. Surface tradeoffs.**

## Project Name
Thai Customer Feedback Analyzer

## Description
An automated NLP pipeline to classify customer sentiment and issue categories from Thai text. The system features a simple batch inference pipeline and an interactive dashboard.

## Explicit Assumptions & Tradeoffs
- **Assumption:** WangchanBERTa is sufficient as a foundation model; no heavier LLMs are required for this specific task.
- **Assumption:** Customers primarily need historical batch analysis (CSV uploads), not real-time streaming APIs.
- **Tradeoff:** We bias toward simple scripts over complex abstractions. Flexibility and configurability that aren't explicitly requested will be excluded.

---

# 2. Problem Statement
Businesses receive thousands of customer comments. Manually categorizing them is slow and unscalable. An automated system is needed to identify sentiment and categorize common business issues. The solution should be implemented with the minimum amount of code necessary.

---

# 3. Scope & Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

## Included Features
- **Sentiment Classification**: Positive, Neutral, Negative.
- **Issue Classification**: Delivery Issue, Product Defect, Product Question, Refund Request.
- **Batch Processing**: Simple CSV in, CSV out pipeline.
- **Analytics Dashboard**: Streamlit UI for static data visualization.

## Excluded Features 
*No features beyond what was asked. Do not add abstractions for single-use code.*
- NO Topic Modeling
- NO Aspect-Based Sentiment Analysis
- NO LLM Summarization
- NO Real-Time Streaming or queues
- NO Multi-Language Support
- NO RAG Systems

---

# 4. Dataset & Annotation Guidelines

## Sentiment Dataset
- **Dataset:** Wisesight Sentiment Corpus (Positive, Neutral, Negative)

## Issue Classification Dataset
- **Dataset:** Custom Annotated Dataset. Target: 300–500 samples per class.
- **Classes & Rules:**
  - **Delivery Issue**: Shipping problems (e.g., ของยังไม่ถึงเลย)
  - **Product Defect**: Broken/malfunctioning (e.g., สินค้าเสียตั้งแต่แกะกล่อง)
  - **Product Question**: Requests for info (e.g., มีสีดำหรือเปล่า)
  - **Refund Request**: Order cancellation (e.g., ขอคืนเงินได้ไหม)

## Label Priority (Avoid Ambiguity)
If multiple labels apply, assign the highest-priority business action:
`Refund Request` -> `Delivery Issue` -> `Product Defect` -> `Product Question`

---

# 5. Tech Stack & Architecture (Simplicity First)

**If you write 200 lines and it could be 50, rewrite it.**

- **Language**: Python 3.12+
- **NLP**: PyTorch, Transformers, PyThaiNLP, WangchanBERTa
- **Dashboard**: Streamlit, Plotly
- **Tracking**: Weights & Biases

## Architecture
Keep the architecture surgical and simple. Avoid over-engineering the data flow.
`Raw Data` -> `Clean Script` -> `WangchanBERTa` -> `Inference Script` -> `Streamlit Dashboard`

---

# 6. Functional Requirements & Goal-Driven Execution

**Define success criteria. Loop until verified.**

## FR-01: CSV Upload & Data Cleaning
- **Objective:** Upload CSV and clean text (remove URLs, duplicates, normalize whitespace).
- **Verify:** `pytest tests/test_preprocessing.py` passes for all edge cases.

## FR-02: Sentiment Prediction
- **Objective:** Predict Positive, Neutral, Negative.
- **Verify:** `predict_sentiment(text)` returns correct schema `(Label, Confidence Score)`.

## FR-03: Issue Prediction
- **Objective:** Predict Delivery, Defect, Question, Refund.
- **Verify:** `predict_issue(text)` returns correct schema `(Label, Confidence Score)`.

## FR-04: Batch Processing
- **Objective:** Process a list of reviews.
- **Verify:** Output CSV matches exact required format (`review_text`, `sentiment`, `sentiment_score`, `issue`, `issue_score`).

## FR-05: Dashboard Visualization
- **Objective:** Render Pie/Bar charts based on predictions.
- **Verify:** Dashboard runs locally and renders up to 5,000 rows correctly.

---

# 7. Non-Functional Requirements & Evaluation Metrics

- **Performance:** Single prediction < 2 seconds.
- **Reproducibility:** All training runs must be triggered via configuration files (e.g., `python train.py --config configs/sentiment.yaml`).

## Goal-Driven Metric Verification
1. **Sentiment Model:** 
   → *verify*: Evaluation loop calculates F1 Score ≥ 85%.
2. **Issue Model:** 
   → *verify*: Evaluation loop calculates Macro F1 Score ≥ 80%.

---

# 8. Folder Structure (Surgical Changes)

**Touch only what you must. Clean up your own mess.**
If development creates unused imports/functions, remove them. Do not include unnecessary files.

```text
thai-customer-feedback-analyzer/
├── configs/          (Strict config files for reproducibility)
├── data/             (Raw & Processed)
├── src/              (Data, Training, Inference, Dashboard scripts)
├── models/           (Saved checkpoints)
├── tests/            (Unit tests for verification loops)
└── pyproject.toml
```

---

# 9. Development Milestones

For multi-step tasks, execute the following loops:

1. **Data Prep** → verify: `data/processed/` contains clean CSVs.
2. **Sentiment Model** → verify: F1 ≥ 85% is logged in W&B and checkpoint saved.
3. **Issue Model** → verify: Macro F1 ≥ 80% is logged in W&B and checkpoint saved.
4. **Inference Pipeline** → verify: passing an input CSV outputs a predictions CSV.
5. **Dashboard** → verify: `streamlit run` launches successfully and loads charts.

---

# 10. Risk Assessment

- **Risk:** Custom issue dataset labeling ambiguity.
- **Mitigation:** Rely strictly on the explicit label priority list.
- **Fallback (Simplicity First):** If the issue model fails to meet the 80% F1 criteria, cut the feature and release a sentiment-only version. Do not build complex workarounds.
