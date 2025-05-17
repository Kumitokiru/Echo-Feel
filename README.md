# EchoFeel – Sentiment Analyzer

**EchoFeel** is a sentiment analysis web app built using Flask and machine learning. It allows users to input product reviews (in English, Tagalog, or Taglish) and receive real-time sentiment predictions: **Negative**, **Neutral**, or **Positive**.

The model is trained on a **combined dataset** from:
- FastText sample review data (`test.ft.txt`)
- Shopee customer reviews (`SHOPEE_REVIEWS.csv`)



## ⚙️ Setup Instructions

### 1. Clone or download the repository

```terminal
git clone https://github.com/Kumitokiru/Echo-Feel.git
```
```terminal
cd echofeel
```

### 2. Install required packages

On your terminal or VS Code's terminal

```terminal
pip install -r requirements.txt
```

### 3. Prepare NLTK resources

These are automatically downloaded by the app, or you can pre-download:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

Dataset Download
The training dataset is not included in the GitHub repository due to size limits.

➡️ Download the ZIP file (~98MB):
Download EchoFeel Training Data
(https://drive.google.com/drive/folders/1wZfOkLcqlOcWS9-Z5zDcKkjyqm4rlBhx?usp=sharing)

After downloading:

Extract the contents into a folder named data/ at the root of the project.

You should end up with:

    
data/
├── test.ft.txt
└── SHOPEE_REVIEWS.csv

Training the Model

Open the train_combined.ipynb notebook and run all cells. This will:

Load and combine FastText + Shopee reviews

Preprocess using English stopwords

Train a 3-class Logistic Regression model

Save the model and TF-IDF vectorizer under models/

Start the web server with:
python app.py


Then visit:
http://127.0.0.1:5000
or visit sraight
https://echo-feel.onrender.com

