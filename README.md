# Email Spam Classification

A Machine Learning pipeline built with Python and `scikit-learn` that classifies text messages as **Spam** or **Ham** (not spam).

## Overview

This project uses the public [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). The `spam_classifier.py` script automatically handles the entire pipeline:
- Downloading and extracting the dataset.
- Preprocessing the text data (lowercasing, removing punctuation, and stripping out common English stopwords).
- Extracting numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).
- Training a `MultinomialNB` (Naive Bayes) classification model.
- Evaluating the model and outputting metrics.

The model achieves an accuracy of approximately **97%** on the test dataset.

## Requirements

- Python 3.x

## Setup & Installation

1. Create a new virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- **Windows**: `.\venv\Scripts\activate`
- **macOS/Linux**: `source venv/bin/activate`

3. Install the required dependencies (Install the requirments.txt file):
```bash
pip install pandas scikit-learn nltk
```

## Usage

Run the main script:
```bash
python spam_classifier.py
```

The script will automatically handle the dataset download, train the model, print the classification report (precision, recall, f1-score), and provide sample predictions.

## Example Output

```text
Loading data...
Dataset shape: (5572, 2)
Preprocessing text...
Extracting features using TF-IDF...
Splitting data into training and test sets...
Training Multinomial Naive Bayes model...
Evaluating model...

==============================
--- Model Evaluation ---
Accuracy: 0.9704

Classification Report:
              precision    recall  f1-score   support

         ham       0.97      1.00      0.98       966
        spam       1.00      0.78      0.88       149

    accuracy                           0.97      1115
   macro avg       0.98      0.89      0.93      1115
weighted avg       0.97      0.97      0.97      1115
==============================

--- Sample Predictions ---
Message: 'Congratulations! You've won a $1000 gift card. Reply YES to claim now.'
Prediction: SPAM

Message: 'Hey, are we still meeting for lunch at 1 PM tomorrow?'
Prediction: HAM
```
