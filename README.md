import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

# ----------------------------
# 1. Load a labeled dataset
# ----------------------------
dataset_url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
df = pd.read_csv(dataset_url)

print(df.head())

# Use only text + sentiment columns
df = df[['label', 'tweet']]
df['label'] = df['label'].map({0: "negative", 1: "neutral", 2: "positive"})

# ----------------------------
# 2. Preprocessing function
# ----------------------------
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@[A-Za-z0-9_]+", '', text)
    text = re.sub(r"#[A-Za-z0-9_]+", '', text)
    text = re.sub(r"[^A-Za-z ]+", '', text)
    text = text.lower()
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

df["clean_tweet"] = df["tweet"].apply(clean_text)

# ----------------------------
# 3. Split data
# ----------------------------
X = df["clean_tweet"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 4. TF-IDF Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# 5. Train Logistic Regression
# ----------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# ----------------------------
# 6. Evaluate
# ----------------------------
pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# ----------------------------
# 7. Predict custom tweet
# ----------------------------
def predict_tweet(tweet):
    cleaned = clean_text(tweet)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)[0]
    return result

# Chat loop
print("\nType a tweet to analyze (or 'exit'): ")
while True:
    t = input("> ")
    if t.lower() == "exit":
        break
    print("Sentiment →", predict_tweet(t))
