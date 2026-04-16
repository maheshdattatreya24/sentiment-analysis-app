import kagglehub
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils import clean_text

# Download dataset
path = kagglehub.dataset_download("kazanova/sentiment140")

file_path = os.path.join(path, "training.1600000.processed.noemoticon.csv")

# Load dataset
df = pd.read_csv(file_path, encoding='latin-1', header=None)
df.columns = ["sentiment", "id", "date", "query", "user", "text"]

# Convert labels
df['sentiment'] = df['sentiment'].replace({0: 0, 4: 1})

# Reduce size for faster training (important)
df = df.sample(n=50000, random_state=42)

# Clean text
df['clean_text'] = df['text'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and Vectorizer saved!")