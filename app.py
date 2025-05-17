from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Only these downloads are needed
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load model & vectorizer
model      = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# English stopwords + lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = nltk.word_tokenize(text)            # now uses punkt only
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# Map labels
LABEL_MAP = {
    0: '❌ Negative',
    1: '😐 Neutral',
    2: '✅ Positive'
}

@app.route('/', methods=['GET','POST'])
def index():
    prediction = None
    if request.method == 'POST':
        review = request.form.get('review_text','').strip()
        if not review:
            prediction = '⚠️ Please enter some text.'
        else:
            clean = preprocess(review)
            vect  = vectorizer.transform([clean])
            pred  = model.predict(vect)[0]
            prediction = LABEL_MAP.get(pred, f'Unknown ({pred})')
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    # Listen on all interfaces & respect PORT (for Render/prod)
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
