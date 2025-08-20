import os
import traceback
import json
import pandas as pd
import re
from flask import Flask, request, jsonify
from docx import Document
import PyPDF2
from flask_cors import CORS
import openai

# ------------------ Config ------------------
FEATURES_FILE = "features.txt"
BASE_PRICE_PER_FEATURE = 800
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

# ------------------ Flask App ------------------
app = Flask(__name__)
CORS(app, origins=["*"])  # Allow all origins for simplicity

# ------------------ Lazy-loaded ML/OpenAI ------------------
clf = None
vectorizer = None
client = None

def get_openai_client():
    global client
    if client is None:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client

# ------------------ Load Features ------------------
def load_features(file_path: str = FEATURES_FILE):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

FEATURES = load_features()

# ------------------ File Reading ------------------
def read_word(file_stream) -> str:
    try:
        doc = Document(file_stream)
        return "\n".join([p.text for p in doc.paragraphs])
    except:
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
    except:
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

# ------------------ OpenAI Feature Suggestion ------------------
def suggest_features_with_openai(content: str):
    try:
        client = get_openai_client()
        prompt = f"""
        Extract a JSON array of software feature names from the following text:
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
    except:
        traceback.print_exc()
        return []

# ------------------ Routes ------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Feature Analyzer API"})

@app.route("/analyze", methods=["POST"])
def analyze_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    content = read_file(file)
    if content is None:
        return jsonify({"error": "Unsupported file type"}), 400

    # Lazy load ML/OpenAI if needed
    features = suggest_features_with_openai(content)

    total_price = sum([calculate_feature_price(f) for f in features])

    # Save results to dataset CSV
    feature_row = {"total_price": total_price}
    for f in features:
        feature_row[f] = calculate_feature_price(f)

    if os.path.exists(DATASET_FILE) and os.path.getsize(DATASET_FILE) > 0:
        df_existing = pd.read_csv(DATASET_FILE)
        df_combined = pd.concat([df_existing, pd.DataFrame([feature_row])], ignore_index=True)
    else:
        df_combined = pd.DataFrame([feature_row])

    df_combined.to_csv(DATASET_FILE, index=False, header=True)

    return jsonify({"features": features, "total_price": total_price})

# ------------------ Run ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Cloud Run expects 8080
    app.run(host="0.0.0.0", port=port)
