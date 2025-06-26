# ie7500-g1-tweet-sentiment-nlp

Course project for **IE7500: Natural Language Processing**  
Titled: **Tweets Sentiment Analysis and Classification**
**Group 1:** Maryam AlBahri, Min Yao, Shan Lin

---

## Project Overview

This project focuses on sentiment classification of tweets using both traditional and deep NLP models.  
We apply and compare **Logistic Regression (TF-IDF)**, **LSTM**, and **BERT** on the Sentiment140 dataset to classify tweets into **positive** and **negative** sentiment categories.


### Objectives

- Compare model performance: **Logistic Regression** vs **LSTM** vs **BERT**
- Understand trade-offs in **speed**, **complexity**, and **performance**
- Build a **robust classifier** for real-world, noisy, short-form text like tweets
- Mitigate **bias** and avoid **data leakage** through careful data handling

---

## Methodology

The modeling workflow is organized as follows:

1. **Exploratory Data Analysis (EDA)**  
   Analyze tweet length, sentiment balance, and text noise (e.g., slang, emojis, elongations).

2. **Data Preprocessing**  
   - Clean raw tweets: lowercasing, removing URLs, token normalization
   - Tokenize and vectorize: TF-IDF for LR; embedding matrix for LSTM; tokenizer for BERT
   - Split dataset into `train`, `validation`, and `test` sets

3. **Model Development**  
   - Train and validate:  
     - **Logistic Regression (TF-IDF baseline)**  
     - **LSTM with embedding layer**  
     - **BERT (transformer fine-tuning)**
   - Evaluate on **validation set** using:
     - Accuracy, Precision, Recall, F1 Score
     - **ROC-AUC** (for final model selection)
   - Perform **manual inference** on handcrafted samples:
     - `"Wow this is amazing"` → Positive  
     - `"This is bad"` → Negative  

4. **Final Model Selection & Retraining**  
   - Select the model with highest ROC-AUC
   - Retrain the selected model using combined **train + validation** data
   - Evaluate final performance on **unseen test set**

---

## Bias Mitigation Strategy

- **Validation data** is strictly used for **model selection** (hypothesis class).
- **Test data** remains untouched during development to prevent leakage.
- Only after final model selection do we retrain on `train + val` and evaluate on the `test` set for a **fair, unbiased** generalization estimate.

---

## Repository Structure

```
.
├── data/ # Raw and external data
├── processed_data/ # Cleaned datasets and splits
│ ├── preprocessed_tweets.csv
│ ├── train_dataset_comp.zip
│ ├── val_dataset.csv
│ ├── test_dataset.csv
│
├── scripts/ # Jupyter Notebooks
│ ├── 1. Exploratory Data Analysis (EDA).ipynb
│ ├── 2. Data Preprocessing.ipynb
│ ├── 3. Model_Development.ipynb
│
├── utils/ # Helper functions
│ └── helper.py
├── requirements.txt # Dependencies
└── README.md # Project overview and guide

```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ie7500-g1-tweet-sentiment-nlp.git (or ssh)
cd ie7500-g1-tweet-sentiment-nlp
```

### 2. Create and activate a virtual environment


python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## License

This repository is intended for academic use only as part of Northeastern University's IE7500 course.
