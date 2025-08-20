import os
import traceback
import json
import joblib
import pandas as pd
import re
from flask import Flask, request, jsonify
from docx import Document
import PyPDF2
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Tuple
from flask_cors import CORS  # <-- import CORS

# ------------------ Config ------------------
FEATURES_FILE = "features.txt"
BASE_PRICE_PER_FEATURE = 800
ML_MODEL_FILE = "feature_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
DATASET_FILE = "features_dataset.csv"

COMPLEXITY_KEYWORDS = {
    "ai": 1200,
    "analytics": 1000,
    "integration": 1000,
    "payment": 900,
    "chatbot": 1100,
    "reporting": 900,
    "security": 1000,
    "ocr": 1200,
    "tracking": 950,
    "forecast": 1000,
}

# ------------------ OpenAI Client ------------------
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "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")
)

# ------------------ Load Features ------------------
def load_features(file_path: str = FEATURES_FILE) -> List[str]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

FEATURES = load_features()

# ------------------ File Reading ------------------
def read_word(file_stream) -> str:
    try:
        doc = Document(file_stream)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception:
        traceback.print_exc()
        return ""

def read_pdf(file_stream) -> str:
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception:
        traceback.print_exc()
        return ""

def read_file(file) -> str:
    ext = os.path.splitext(file.filename)[1].lower()
    if ext == ".docx":
        return read_word(file)
    elif ext == ".pdf":
        return read_pdf(file)
    return None

# ------------------ Feature Pricing ------------------
def calculate_feature_price(feature: str) -> int:
    price = BASE_PRICE_PER_FEATURE
    for keyword, extra in COMPLEXITY_KEYWORDS.items():
        if keyword.lower() in feature.lower():
            price += extra
    return price

def detect_features(content: str) -> Tuple[Dict[str, Dict[str, int]], int]:
    detected = {}
    total_price = 0
    for feature in FEATURES:
        if feature.lower() in content.lower():
            price = calculate_feature_price(feature)
            detected[feature] = {"price": price}
            total_price += price
    return detected, total_price

# ------------------ OpenAI Feature Suggestion ------------------
def suggest_features_with_openai(content: str) -> List[str]:
    try:
        prompt = f"""
        Read the following document content and extract a list of suggested software features.
        Return only the feature names in a JSON array.
        Document content:
        {content}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        return [str(f) for f in json.loads(match.group())] if match else []
    except Exception:
        traceback.print_exc()
        return []

# ------------------ ML Model ------------------
def train_ml_model(dataset_file: str = DATASET_FILE):
    if not os.path.exists(dataset_file):
        return None, None

    # Read CSV with explicit header
    df = pd.read_csv(dataset_file)
    if 'document_text' not in df.columns:
        print("Dataset missing 'document_text' column")
        return None, None

    # Fill missing values in feature columns
    df = df.fillna(0)

    # Separate X and y
    X = df['document_text']
    y = df.drop(columns=['document_text'])

    # Ensure column names are strings and clean any accidental numeric headers
    y.columns = [str(col).strip() for col in y.columns]

    # Fit vectorizer
    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)

    # Fit multi-output classifier
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    clf.fit(X_vect, y)

    # Save model and vectorizer
    joblib.dump(clf, ML_MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    return clf, vectorizer


# Load model if exists
clf, vectorizer = None, None
if os.path.exists(ML_MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    clf = joblib.load(ML_MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)


def predict_features_ml(content: str) -> List[str]:
    if clf is None or vectorizer is None:
        return []

    # Transform content using vectorizer
    vect = vectorizer.transform([content])

    # Predict features
    prediction = clf.predict(vect)
    predicted_features = []

    # Map predictions to proper feature names
    for i, val in enumerate(prediction[0]):
        if val == 1:
            predicted_features.append(str(clf.classes_[i]))

    # Remove any accidental numeric placeholders like "[1.]"
    predicted_features = [f for f in predicted_features if not re.match(r'^\[?\d+\.?\]?$', f)]

    return predicted_features

# ------------------ Flask App ------------------
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the Feature Analyzer API",
        "endpoints": {
            "/": "gg nothing "
        }
    })
@app.route("/analyze", methods=["POST"])
def analyze_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Read file content
    content = read_file(file)
    if content is None:
        return jsonify({"error": "Unsupported file type. Only .docx and .pdf are supported."}), 400

    # OpenAI suggested features
    openai_features = suggest_features_with_openai(content)

    # Calculate total price
    total_price = sum([calculate_feature_price(f) for f in openai_features])

    # Save only OpenAI features + total_price to dataset CSV
    feature_row = {"total_price": total_price}
    for feature in openai_features:
        feature_row[feature] = calculate_feature_price(feature)

    if os.path.exists(DATASET_FILE) and os.path.getsize(DATASET_FILE) > 0:
        df_existing = pd.read_csv(DATASET_FILE)
        df_combined = pd.concat([df_existing, pd.DataFrame([feature_row])], ignore_index=True)
    else:
        df_combined = pd.DataFrame([feature_row])

    df_combined.to_csv(DATASET_FILE, index=False, header=True)

    return jsonify({
        "openai_features": openai_features,
        "total_price": total_price
    })
if __name__ == "__main__":
    app.run(debug=True, port=7500)
