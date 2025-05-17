from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# -------------------------------------------------------------------
# 1. NLTK Setup: Download required data on first run
nltk.download('punkt')  # Tokenizer models
nltk.download('stopwords')  # Stopword lists
nltk.download('wordnet')    # Lemmatizer data
nltk.download('punkt_tab')  # Optional tokenizer variant

# 2. Initialize Flask application
app = Flask(__name__)

# -------------------------------------------------------------------
# 3. Load pre-trained artifacts
#    - 'model' for inference
#    - 'vectorizer' for text feature extraction
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# -------------------------------------------------------------------
# 4. Setup NLP utilities
stop_words = set(stopwords.words('english'))    # English stopwords
lemmatizer = WordNetLemmatizer()    # WordNet lemmatizer

# 5. Text preprocessing function
def preprocess(text):
    """
    Clean and normalize input text:
      1. Lowercase conversion
      2. HTML tag removal
      3. Punctuation stripping
      4. Tokenization
      5. Stopword removal
      6. Lemmatization
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)            # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)     # Remove punctuation
    tokens = nltk.word_tokenize(text)              # Tokenize
    tokens = [
        lemmatizer.lemmatize(t) for t in tokens
        if t not in stop_words                      # Filter stopwords
    ]
    return " ".join(tokens)                        # Return cleaned text

# -------------------------------------------------------------------
# 6. Mapping from numeric class to human-readable label
LABEL_MAP = {
    0: '❌ Negative',
    1: '😐 Neutral',
    2: '✅ Positive'
}

# -------------------------------------------------------------------
# 7. Define Flask route for sentiment analysis
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Get the user’s review text from the form
        review = request.form.get('review_text', '').strip()

        if not review:
            # If no input, prompt the user
            prediction = '⚠️ Please enter some text.'
        else:
            # Preprocess, vectorize, predict, and map to label
            clean      = preprocess(review)
            vect       = vectorizer.transform([clean])
            pred_class = model.predict(vect)[0]
            prediction = LABEL_MAP.get(pred_class, f'Unknown ({pred_class})')

    # Render the HTML template with the prediction result
    return render_template('index.html', prediction=prediction)

# -------------------------------------------------------------------
# 8. Run the Flask development server
if __name__ == '__main__':
    # Use PORT env var if available (for cloud), else default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
