import os
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download dataset
path = kagglehub.dataset_download("hijest/genre-classification-dataset-imdb")
print("Dataset Path:", path)

# Find training file
train_file = None
for file in os.listdir(path):
    if "train" in file.lower() and file.endswith(".txt"):
        train_file = os.path.join(path, file)
        break

if train_file is None:
    raise FileNotFoundError("Training file not found!")

# Read dataset
data = []
with open(train_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.split(" ::: ")
        if len(parts) >= 4:
            genre = parts[1]
            description = parts[3].strip()
            data.append([genre, description])

df = pd.DataFrame(data, columns=["Genre", "Description"])

# Features and labels
X = df["Description"]
y = df["Genre"]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
