# Simple Emotion Detection Test
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

print("🚀 Starting Simple Emotion Detection...")

# Sample data
data = {
    'text': [
        "I am happy today", "This is wonderful", "I feel great",
        "I am sad", "This is terrible", "I feel bad",
        "I am angry", "This makes me mad", "I hate this"
    ],
    'emotion': ['happy', 'happy', 'happy', 
                'sad', 'sad', 'sad', 
                'angry', 'angry', 'angry']
}

df = pd.DataFrame(data)

# Simple text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['emotion']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Test
accuracy = model.score(X, y)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")

# Prediction function
def predict_emotion(text):
    cleaned = clean_text(text)
    text_vector = vectorizer.transform([cleaned])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector).max()
    
    print(f"📝 Text: '{text}'")
    print(f"🎯 Emotion: {prediction}")
    print(f"📊 Confidence: {probability:.2%}")
    return prediction

# Test examples
print("\n🧪 Testing the model:")
predict_emotion("I am so happy!")
predict_emotion("This is terrible")
predict_emotion("I feel angry")

print("\n🎉 Emotion Detection is working!")