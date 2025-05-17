from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# 1. Ensure NLTK data is downloaded (first run only)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# 2. Load the trained sentiment model and TF-IDF vectorizer
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# 3. Load English stopwords
stop_words = set(stopwords.words('english'))

# 4. Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# 5. Preprocessing function (same as training)
def preprocess(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)              # remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)       # remove punctuation
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# 6. Mapping from class label to human-readable label
LABEL_MAP = {
    0: '❌ Negative',
    1: '😐 Neutral',
    2: '✅ Positive'
}

# 7. Route for form and prediction
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        review = request.form.get('review_text', '').strip()
        if not review:
            prediction = '⚠️ Please enter some text.'
        else:
            clean = preprocess(review)
            vect = vectorizer.transform([clean])
            pred = model.predict(vect)[0]
            prediction = LABEL_MAP.get(pred, f'Unknown ({pred})')
    return render_template('index.html', prediction=prediction)

# 8. Start the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # fallback to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=True)

