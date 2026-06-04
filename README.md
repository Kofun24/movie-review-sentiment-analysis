# Movie Review Sentiment Analysis

This is a machine learning project that predicts whether a movie review is positive or negative.

## Project Overview

The aim of this project is to build a sentiment analysis model using Natural Language Processing and Machine Learning. The model is trained using IMDB movie reviews.

## Problem Statement

Movie reviews are written in natural language. The goal of this project is to classify each movie review into one of two categories:

- Positive
- Negative

## Dataset

The dataset contains IMDB movie reviews.

| Column | Description |
|---|---|
| review | Movie review text |
| sentiment | Sentiment label: positive or negative |

## Project Workflow

1. Load dataset
2. Exploratory Data Analysis
3. Text preprocessing
4. TF-IDF vectorization
5. Model training
6. Model evaluation
7. Best model selection
8. Save trained model
9. Build Streamlit app

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- WordCloud
- Joblib
- Streamlit

## Machine Learning Models

The following models were trained and compared:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM
4. Random Forest

## Project Structure

```text
Movie-Review-Sentiment-Analysis/
│
├── data/
│   ├── imdb_reviews.csv
│   └── imdb_cleaned.csv
│
├── models/
│   └── movie_sentiment_pipeline.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_text_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── app.py
├── predict.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore