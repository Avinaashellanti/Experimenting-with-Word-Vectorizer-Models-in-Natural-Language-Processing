import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# OPTION 1: Load dataset from online link
url = "https://raw.githubusercontent.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k/master/IMDB-Dataset.csv"
df = pd.read_csv(url)

# Convert labels
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Preprocessing
df["text"] = df["review"].str.lower()

# Take subset (same as your PDF)
df = df.sample(10000, random_state=42).reset_index(drop=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42
)

# Function to run models
def run_model(vectorizer, model, vec_name, model_name):
    start = time.time()
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    end = time.time()
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"{vec_name} | {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Time: {end-start:.2f}s\n")

# Vectorizers
bow = CountVectorizer(stop_words='english', max_features=10000)
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)

# Models
lr = LogisticRegression(max_iter=200)
svm = LinearSVC()

# Run experiments
run_model(bow, lr, "BoW", "Logistic Regression")
run_model(tfidf, lr, "TF-IDF", "Logistic Regression")
run_model(tfidf, svm, "TF-IDF", "SVM")
