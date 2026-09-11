# Sentiment Analysis on App Reviews using Lexicon-Based Labeling + IndoBERT

A sentiment analysis pipeline that combines lexicon-based auto-labeling with a fine-tuned **IndoBERT** transformer model to classify user review sentiment (positive/negative).

## 🎯 Overview

Raw app reviews often come with no sentiment labels. This project tackles that by:
1. **Auto-labeling** reviews using a lexicon-based approach (VADER) as an initial labeling strategy
2. **Fine-tuning IndoBERT** (`indolem/indobert-base-uncased`) on those labels to build a robust sentiment classifier

This was originally built during a Data Analyst internship to analyze user sentiment from app reviews. The dataset used in the original work is proprietary and not included here — the notebook is shared to document the methodology, code, and results.

## 📊 Results

**Label mapping:** `0` = negative, `1` = positive.

| Metric | Score |
|---|---|
| Accuracy | 99.22% |
| Precision | 99.23% |
| Recall | 99.22% |
| F1-score | 99.21% |
| Eval Loss | 0.0589 |

The confusion matrix below follows the same mapping — rows/columns labeled `0` refer to **negative**, and `1` refers to **positive**.

The model achieved very high classification performance using a **single-pipeline** approach, where lexicon labels are generated on the fly rather than being saved as a separate intermediate dataset. In a separate experiment (two-stage pipeline, where lexicon labels are saved before training), accuracy was slightly lower (98.05%) — suggesting the intermediate save/reload step introduced some label noise. This indicates end-to-end pipelines can outperform staged approaches in maintaining data quality for this task.

## 🧠 Methodology

```
Raw text reviews
      │
      ▼
Text cleaning (lowercase, remove URLs/mentions/hashtags/punctuation)
      │
      ▼
Lexicon-based auto-labeling (VADER) → positive / negative / neutral
      │
      ▼
Drop neutral, keep binary positive/negative labels
      │
      ▼
Train/test split (80/20, stratified)
      │
      ▼
Fine-tune IndoBERT (5 epochs, lr=2e-5)
      │
      ▼
Evaluate: accuracy, precision, recall, F1, confusion matrix
```

## 🛠️ Tech Stack

- **Model:** IndoBERT (`indolem/indobert-base-uncased`) via HuggingFace `transformers`
- **Auto-labeling:** NLTK VADER SentimentIntensityAnalyzer
- **Training:** HuggingFace `Trainer` API
- **Evaluation & visualization:** scikit-learn, matplotlib, seaborn

## 📁 Repository Structure

```
├── sentiment_analysis_indobert.ipynb   # Main notebook (clean, end-to-end pipeline)
├── results/                            # Evaluation charts (confusion matrix, metrics)
└── README.md
```

## 🚀 Running This Notebook

1. Clone this repo and open `sentiment_analysis_indobert.ipynb` in Jupyter or Google Colab
2. Replace the data loading step with your own labeled text dataset (a CSV with a text column)
3. Run all cells sequentially

```bash
pip install transformers accelerate nltk scikit-learn pandas matplotlib seaborn
```

## 📌 Notes

- The original dataset is confidential client data and has been excluded from this repository in accordance with data privacy practices.
- This notebook uses a binary (positive/negative) classification setup; neutral-labeled data is dropped after the lexicon labeling step.

## 👤 Author

**Muhammad Hafidlul Qolbi**
Informatics Engineering, UIN Maulana Malik Ibrahim Malang
[LinkedIn](https://linkedin.com/in/hafidlul-qolbi) · [GitHub](https://github.com/hafidqolbi)
