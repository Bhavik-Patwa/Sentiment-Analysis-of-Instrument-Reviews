# Sentiment Analysis of Musical Instrument Reviews


## Project Overview

This project implements an end-to-end sentiment analysis system for musical instrument reviews. Using natural language processing (NLP) and supervised machine learning, the system classifies customer reviews into three sentiment categories : positive, neutral and negative. After evaluating multiple models, a tuned Support Vector Machine (SVM) achieves **99.7% accuracy** on a balanced test set, making it suitable for deployment.

The complete pipeline includes text preprocessing, feature extraction via TF-IDF, handling class imbalance with SMOTE, training several classifiers, hyperparameter tuning and comprehensive evaluation.

## Problem Statement

Businesses rely on customer feedback to improve products and services, but manually analyzing thousands of reviews is impractical. An automated sentiment classifier can quickly extract actionable insights from product reviews. This project focuses specifically on musical instrument reviews, a domain with distinct vocabulary and sentiment patterns.

## Project Objectives

- Clean and preprocess raw review text (tokenization, stopword removal, lemmatization).
- Convert text into numerical features using TF-IDF and CountVectorizer.
- Address class imbalance (majority positive reviews) using SMOTE.
- Train and compare multiple machine learning models : Logistic Regression, SVM, Decision Tree, Random Forest, Naive Bayes, KNN.
- Optimize top-performing models via GridSearchCV.
- Evaluate models using accuracy, precision, recall and F1-score.
- Visualize results with confusion matrices and feature importance plots.
- Demonstrate real-world usage with example predictions.

## Dataset Description

- **Data source** : Amazon musical instrument reviews (file : `data/Instruments_Reviews.csv`).
- **Size** : 10,261 reviews.
- **Key features** :
  - `reviewText` : Full text of the review.
  - `summary` : Short summary of the review.
  - `overall` : Numerical rating from 1 to 5.
  - Additional metadata : reviewerID, asin, reviewerName, helpful votes, timestamps.
- **Target variable** : Sentiment derived from `overall` rating :
  - Positive : rating ≥ 4 (9,022 reviews)
  - Neutral : rating = 3 (772 reviews)
  - Negative : rating ≤ 2 (467 reviews)
- **Timeframe** : Not specified in the dataset; reviews span multiple dates.

## System Architecture / Pipeline

The system follows a standard machine learning pipeline :

1. **Data Ingestion** : Load CSV, handle missing values.
2. **Text Preprocessing** : Combine `reviewText` and `summary`, clean text, remove punctuation, tokenize, remove stopwords, lemmatize.
3. **Feature Engineering** : Transform cleaned text into TF-IDF vectors (5000 features).
4. **Class Balancing** : Apply SMOTE to oversample minority classes (neutral, negative) in the feature space.
5. **Train/Test Split** : 80/20 stratified split after SMOTE.
6. **Model Training** : Train six baseline classifiers.
7. **Hyperparameter Tuning** : Use GridSearchCV on top models (SVM, Random Forest, Logistic Regression).
8. **Evaluation** : Compute metrics and plot confusion matrices.
9. **Interpretation** : Extract feature importance (Random Forest) and test with sample reviews.

## Feature Engineering

- **Text Cleaning** : Lowercasing, removal of punctuation, tokenization, stopword removal and lemmatization reduce noise and dimensionality.
- **TF-IDF Vectorization** : Converts text into a matrix of term frequency–inverse document frequency. Max features set to 5000 to limit dimensionality while retaining informative words.
- **SMOTE** : Synthetic Minority Over-sampling Technique creates synthetic samples for neutral and negative classes to balance the training set, preventing model bias toward the majority positive class.

## Modeling Approach

### Algorithms Used
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Multinomial Naive Bayes
- K-Nearest Neighbors

### Training Strategy
- All models trained on the SMOTE-resampled training set.
- Baseline models evaluated without tuning.
- Top performers (SVM, Random Forest, Logistic Regression) selected for hyperparameter tuning.

### Hyperparameter Tuning
- **SVM** : Grid search over `C` [0.1, 1, 10] and `kernel` ['linear', 'rbf']. Best : `C=10`, `kernel='rbf'`.
- **Random Forest** : Grid search over `n_estimators` [100, 200], `max_depth` [None, 10, 20], `min_samples_split` [2, 5]. Best : `n_estimators=200`, `max_depth=None`, `min_samples_split=2`.
- **Logistic Regression** : Grid search over `C` (logspace from 1e-4 to 1e4) and `penalty` ['l1', 'l2']. Best : `C≈6866`, `penalty='l2'`.

### Cross-Validation
- 5-fold cross-validation used inside GridSearchCV to select optimal hyperparameters.

## Evaluation Strategy

- **Metrics** : Accuracy, precision, recall, F1-score (macro and weighted).
- **Validation Design** : After SMOTE, the dataset is split into 80% training and 20% testing, preserving class distribution in the test set.
- **Results Summary** :

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-------|----------|-------------------|----------------|------------|
| SVM (tuned) | **0.9970** | 0.99 | 0.99 | 0.99 |
| Random Forest (tuned) | 0.9762 | 0.98 | 0.98 | 0.98 |
| Logistic Regression (tuned) | 0.9706 | 0.97 | 0.97 | 0.97 |
| Baseline SVM | 0.9928 | 0.99 | 0.99 | 0.99 |
| Baseline Random Forest | 0.9751 | 0.98 | 0.98 | 0.98 |
| Baseline Logistic Regression | 0.9422 | 0.94 | 0.94 | 0.94 |

The tuned SVM misclassifies only a handful of neutral and negative instances, achieving near-perfect performance.

## Project Structure

```
.
├── data/
│   └── Instruments_Reviews.csv          # Raw dataset
├── notebook/
│   └── Sentiment Analysis of Instruments' Reviews.ipynb   # Main analysis notebook
└── README.md                              # This file
```

## Tech Stack

- **Python 3.12**
- **Data manipulation** : pandas, numpy
- **NLP preprocessing** : NLTK (punkt, stopwords, wordnet), regex
- **Machine learning** : scikit-learn (models, feature extraction, SMOTE, GridSearchCV)
- **Imbalanced learning** : imbalanced-learn (SMOTE)
- **Visualization** : matplotlib, seaborn, wordcloud
- **Environment** : Jupyter Notebook

## Installation & Setup

1. Clone the repository :
   ```bash
   git clone https://github.com/Bhavik-Patwa/Sentiment-Analysis-of-Instrument-Reviews.git
   cd Sentiment-Analysis-of-Instrument-Reviews
   ```

2. Create a virtual environment (optional but recommended) :
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows : venv\Scripts\activate
   ```

3. Install required packages :
   ```bash
   pip install pandas numpy nltk matplotlib wordcloud imbalanced-learn scikit-learn seaborn
   ```

4. Launch Jupyter Notebook :
   ```bash
   jupyter notebook
   ```
   Open `notebook/Sentiment Analysis of Instruments' Reviews.ipynb`.

## How to Run the Pipeline

All steps are contained within the Jupyter notebook. Execute cells sequentially :

1. **Environment Setup** : Install and import libraries.
2. **Data Loading** : Load `Instruments_Reviews.csv` from the `data/` directory.
3. **Preprocessing** : Handle missing values, combine text columns, clean and lemmatize text.
4. **Feature Extraction** : TF-IDF vectorization (5000 features).
5. **Class Balancing** : Apply SMOTE to training data.
6. **Train/Test Split** : 80/20 split after SMOTE.
7. **Model Training** : Run baseline models.
8. **Hyperparameter Tuning** : Grid search for SVM, Random Forest, Logistic Regression.
9. **Evaluation** : Print classification reports and plot confusion matrices.
10. **Feature Importance** : Visualize top words from Random Forest.
11. **Example Predictions** : Test tuned SVM on new sample reviews.

## Example Output / Results

The system outputs :

- **Classification reports** for all models, showing precision, recall and F1 per class.
- **Confusion matrices** for tuned models (Logistic Regression, SVM, Random Forest).
- **Feature importance plot** for Random Forest (top 20 words).
- **Example predictions** on user-defined sample reviews, e.g. :

```
Review  : "I absolutely love this instrument! The sound quality is amazing."
 Predicted Sentiment  : positive
Review  : "The product is okay, but I expected better durability."
 Predicted Sentiment  : neutral
Review  : "Terrible experience! The item broke within a week."
 Predicted Sentiment  : negative
```

The final model (tuned SVM) achieves **99.7% accuracy** on the test set, correctly classifying almost all instances.

## Future Improvements

- **Deep Learning** : Experiment with LSTM, GRU or Transformer-based models (e.g. BERT) to capture contextual nuances.
- **Class-specific analysis** : Investigate why neutral reviews are sometimes misclassified; possibly improve by adding n-gram features.
- **Deployment** : Wrap the model as a REST API using Flask or FastAPI for real-time sentiment prediction.
- **Feature expansion** : Include metadata (helpful votes, reviewer history) as additional features.
- **Cross-domain testing** : Evaluate model on other product categories to gauge generalizability.
