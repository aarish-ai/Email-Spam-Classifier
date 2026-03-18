import os
import urllib.request
import zipfile
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import string
from nltk.corpus import stopwords
import ssl

# Bypass SSL verification for nltk download if necessary
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords', quiet=True)

def download_and_load_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "smsspamcollection.zip")
    data_path = os.path.join(data_dir, "SMSSpamCollection")
    
    if not os.path.exists(data_path):
        print("Downloading SMS Spam Collection dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Dataset downloaded and extracted.")
    
    # Load dataset
    # SMS Spam Collection is tab-separated with 'label' and 'message' columns
    df = pd.read_csv(data_path, sep='\t', names=['label', 'message'])
    return df

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def main():
    print("Loading data...")
    df = download_and_load_data()
    print(f"Dataset shape: {df.shape}")
    
    print("Preprocessing text...")
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    print("Extracting features using TF-IDF...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_message'])
    y = df['label']
    
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*30)
    print("--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("="*30 + "\n")
    
    # Test with some examples
    sample_messages = [
        "Congratulations! You've won a $1000 gift card. Reply YES to claim now.",
        "Hey, are we still meeting for lunch at 1 PM tomorrow?",
        "URGENT! Your account has been locked. Click here to verify your identity.",
        "Can you send me the report by 5 PM today?"
    ]
    print("--- Sample Predictions ---")
    for msg in sample_messages:
        processed = preprocess_text(msg)
        features = vectorizer.transform([processed])
        prediction = model.predict(features)[0]
        print(f"Message: '{msg}'")
        print(f"Prediction: {prediction.upper()}\n")

if __name__ == "__main__":
    main()
