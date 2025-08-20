# train_model.py
# Install required packages:
# pip install pandas scikit-learn joblib

import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# ------------------ Configuration ------------------
DATASET_FILE = "features_dataset.csv"
ML_MODEL_FILE = "feature_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

def train_ml_model(dataset_file=DATASET_FILE):
    if not os.path.exists(dataset_file):
        print(f"⚠️ Dataset not found: {dataset_file}")
        return None, None
    
    df = pd.read_csv(dataset_file)
    
    if 'document_text' not in df.columns:
        print("❌ Dataset must contain a 'document_text' column")
        return None, None
    
    X = df['document_text']
    y = df.drop(columns=['document_text'])
    
    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)
    
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    clf.fit(X_vect, y)
    
    joblib.dump(clf, ML_MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    
    print("✅ ML model trained and saved")
    return clf, vectorizer

if __name__ == "__main__":
    train_ml_model()
