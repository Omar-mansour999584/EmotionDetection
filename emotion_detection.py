# EmotionDetection# Emotion Detection - Enhanced Version with More Data
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re

print("🚀 Starting Enhanced Emotion Detection Project...")
print("📊 Using 72 samples for better accuracy!")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Create LARGER dataset (72 samples - 12 for each emotion)
data = {
    'text': [
        # Happy emotions (12 samples)
        "I am so happy today!", "This is wonderful", "I feel amazing",
        "I'm thrilled with this result", "What a fantastic day", "I love this",
        "This makes me so happy", "I'm overjoyed", "Feeling great today",
        "This is absolutely wonderful", "I'm so excited", "This is perfect",
        
        # Sad emotions (12 samples)
        "I am very sad today", "This is terrible news", "I feel depressed",
        "I'm so disappointed", "This breaks my heart", "I feel miserable",
        "This is so upsetting", "I'm feeling down", "This makes me unhappy",
        "I feel so sad right now", "This is heartbreaking", "I'm devastated",
        
        # Angry emotions (12 samples)
        "I am angry now", "This makes me furious", "I hate this",
        "I'm so mad about this", "This is infuriating", "I feel outraged",
        "This makes me so angry", "I'm really pissed off", "This is unacceptable",
        "I feel enraged", "This irritates me", "I'm boiling with anger",
        
        # Fear emotions (12 samples)
        "I feel scared", "This is frightening", "I am terrified",
        "I'm afraid of what might happen", "This scares me", "I feel anxious",
        "This is terrifying", "I'm worried about this", "This makes me nervous",
        "I feel threatened", "This is alarming", "I'm panicking",
        
        # Surprise emotions (12 samples)
        "What a surprise!", "This is amazing", "Wow incredible",
        "I'm shocked by this news", "This is unbelievable", "What an unexpected turn",
        "This is astonishing", "I'm stunned", "This caught me off guard",
        "What a shock", "This is incredible", "I'm amazed",
        
        # Neutral emotions (12 samples)
        "I feel neutral about this", "This is okay", "Not bad not good",
        "I'm indifferent", "This is average", "Nothing special",
        "It's fine", "This is normal", "I don't feel strongly",
        "This is mediocre", "It's acceptable", "This is ordinary"
    ],
    'emotion': [
        'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 
        'happy', 'happy', 'happy', 'happy', 'happy', 'happy',
        
        'sad', 'sad', 'sad', 'sad', 'sad', 'sad',
        'sad', 'sad', 'sad', 'sad', 'sad', 'sad',
        
        'angry', 'angry', 'angry', 'angry', 'angry', 'angry',
        'angry', 'angry', 'angry', 'angry', 'angry', 'angry',
        
        'fear', 'fear', 'fear', 'fear', 'fear', 'fear',
        'fear', 'fear', 'fear', 'fear', 'fear', 'fear',
        
        'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise',
        'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise',
        
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral'
    ]
}

df = pd.DataFrame(data)
df['cleaned_text'] = df['text'].apply(clean_text)

print("✅ Enhanced dataset created successfully!")
print(f"📊 Total samples: {len(df)}")
print(f"🎯 Emotions: {df['emotion'].unique()}")

# Show emotion distribution
print("\n📈 Emotion Distribution:")
print(df['emotion'].value_counts())

# Prepare features and labels
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['emotion']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Training samples: {X_train.shape[0]}")
print(f"📊 Testing samples: {X_test.shape[0]}")

# Train model
print("\n🧠 Training enhanced model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Enhanced model trained successfully!")
print(f"📈 Accuracy: {accuracy:.2%}")

# Show detailed results
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Prediction function
def predict_emotion(text):
    """
    Predict the emotion of input text
    """
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Transform to features
    text_vector = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    # Get confidence scores
    emotion_probs = {}
    for i, emotion in enumerate(model.classes_):
        emotion_probs[emotion] = probabilities[i]
    
    # Display results
    print(f"\n📝 Input: '{text}'")
    print(f"🎯 Predicted Emotion: {prediction.upper()}")
    print("📊 Top 3 Confidence Scores:")
    
    # Sort by probability and show top 3
    sorted_probs = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    for emotion, prob in sorted_probs:
        print(f"   {emotion}: {prob:.2%}")
    
