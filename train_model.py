import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import time

print("Loading dataset...")
start_time = time.time()
df = pd.read_csv("WELFake_Dataset.csv")
print(f"Loaded {len(df)} rows in {time.time() - start_time:.2f} seconds.")

# Clean data
print("Cleaning data...")
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')
# Combine title and text
df['content'] = df['title'] + " " + df['text']

# The dataset labels are: 1 (Fake) and 0 (Real) in some versions, or vice versa.
# Let's verify by just printing the distribution.
print("Label distribution:")
print(df['label'].value_counts())

X = df['content']
y = df['label']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Vectorizing text (this might take a minute)...")
vectorizer = TfidfVectorizer(max_features=25000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate
accuracy = model.score(X_test_tfidf, y_test)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

print("Saving model and vectorizer...")
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)
    
with open("fake_news_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Done! Model saved as fake_news_model.pkl and fake_news_vectorizer.pkl.")
