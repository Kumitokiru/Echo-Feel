from flask import Flask, render_template, request
import os
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1) Ensure NLTK data is available (only needed on first run) ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# --- 2) Initialize Flask ---
app = Flask(__name__)

# --- 3) Load trained model & vectorizer ---
MODEL_PATH = 'models/sentiment_model.pkl'
VECT_PATH  = 'models/vectorizer.pkl'
model      = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

# --- 4) Build stopwords set & lemmatizer ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- 5) Preprocessing function ---
def preprocess(text: str) -> str:
    """
    1. Lowercase
    2. Strip HTML tags
    3. Remove non-alphanumeric chars
    4. Tokenize
    5. Remove stopwords
    6. Lemmatize
    """
    txt = text.lower()
    txt = re.sub(r"<[^>]+>", "", txt)
    txt = re.sub(r"[^a-zA-Z0-9\s]", "", txt)
    tokens = nltk.word_tokenize(txt)
    clean_tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words
    ]
    return " ".join(clean_tokens)

# --- 6) Label mapping ---
LABEL_MAP = {
    0: '❌ Negative',
    1: '😐 Neutral',
    2: '✅ Positive'
}

# --- 7) Flask route ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        review = request.form.get('review_text', '').strip()
        if not review:
            prediction = '⚠️ Please enter some text.'
        else:
            # Preprocess & vectorize
            clean_text = preprocess(review)
            vect        = vectorizer.transform([clean_text])
            # Predict & map label
            pred        = model.predict(vect)[0]
            prediction  = LABEL_MAP.get(pred, f'Unknown ({pred})')
    return render_template('index.html', prediction=prediction)

# --- 8) Entry point ---
if __name__ == '__main__':
    # Bind to PORT if provided (for cloud deployment), else default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
