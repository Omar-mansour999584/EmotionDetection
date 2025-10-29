# FINAL EMOTION DETECTOR - HIGH ACCURACY VERSION
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import joblib

print("🎯 FINAL EMOTION DETECTION PROJECT")
print("=" * 50)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Create OPTIMIZED dataset with clear emotion words
def create_optimized_dataset():
    emotions_data = {
        'happy': [
            "i am happy", "very happy", "so happy", "feeling happy", "happy today",
            "joy joy joy", "love love love", "amazing fantastic", "wonderful great",
            "excellent perfect", "delighted thrilled", "ecstatic overjoyed",
            "joyful cheerful", "blissful content", "smiling laughing", "good great",
            "best day ever", "awesome superb", "fantastic amazing", "perfect excellent",
            "happy smile", "laughing fun", "enjoying pleasure", "happy heart",
            "sunshine happy", "happy dance", "celebrating happy", "happy moment"
        ],
        'sad': [
            "i am sad", "very sad", "so sad", "feeling sad", "sad today", 
            "unhappy miserable", "depressed down", "sorrow grief", "heartbroken crying",
            "tears sad", "upset disappointed", "hopeless despair", "gloomy bleak",
            "melancholy blue", "dejected downcast", "despondent discouraged",
            "forlorn wretched", "dismal grim", "tragic heartbreaking", "painful sad",
            "hurt suffering", "lonely alone", "abandoned rejected", "broken sad",
            "defeated lost", "weeping mourning", "grieving lamenting", "sadness pain"
        ],
        'angry': [
            "i am angry", "very angry", "so angry", "feeling angry", "angry now",
            "mad furious", "enraged irate", "outraged incensed", "fuming seething",
            "infuriated livid", "wrathful cross", "annoyed irritated", "aggravated vexed",
            "provoked riled", "resentful bitter", "hostile aggressive", "frustrated upset",
            "pissed off", "boiling mad", "seeing red", "hot tempered", "angry rage",
            "hate dislike", "despise loathe", "detest abhor", "angry face", "angry words"
        ],
        'fear': [
            "i am afraid", "very afraid", "so scared", "feeling fear", "fearful now",
            "terrified frightened", "panicked alarmed", "anxious worried", "nervous uneasy",
            "apprehensive hesitant", "timid shy", "cowardly fearful", "intimidated threatened",
            "horrified shocked", "petrified stunned", "startled surprised", "fear dread",
            "phobia anxiety", "worry concern", "fright terror", "panic attack",
            "scared shitless", "afraid worried", "nervous anxious", "tense stressed"
        ],
        'surprise': [
            "i am surprised", "very surprised", "so surprised", "feeling surprise", "surprised now",
            "shocked amazed", "astonished stunned", "astounded bewildered", "flabbergasted dumbfounded",
            "startled taken aback", "unexpected sudden", "unanticipated unforeseen", "surprise shock",
            "amazement wonder", "astonishment awe", "incredible unbelievable", "wow amazing",
            "oh my god", "no way really", "unbelievable incredible", "surprise party",
            "pleasant surprise", "surprise gift", "surprise visit", "surprise event"
        ],
        'neutral': [
            "i am neutral", "feeling neutral", "neutral mood", "neutral now",
            "okay fine", "alright acceptable", "average normal", "ordinary regular",
            "standard typical", "usual common", "moderate medium", "balanced even",
            "calm peaceful", "quiet still", "nothing special", "not bad", "not good",
            "so so", "meh whatever", "no strong feelings", "no opinion", "no reaction",
            "blank empty", "void neutral", "indifferent unconcerned", "apathetic uninterested"
        ]
    }
    
    texts = []
    emotions = []
    for emotion, emotion_texts in emotions_data.items():
        texts.extend(emotion_texts)
        emotions.extend([emotion] * len(emotion_texts))
    
    return pd.DataFrame({'text': texts, 'emotion': emotions})

# Create dataset
df = create_optimized_dataset()
df['cleaned_text'] = df['text'].apply(clean_text)

print(f"📊 Training with {len(df)} samples")
print(f"🎯 Emotions: {set(df['emotion'])}")

# Optimized vectorizer
vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 1),
    stop_words='english'
)

X = vectorizer.fit_transform(df['cleaned_text'])
y = df['emotion']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model with better parameters
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    C=1.0,
    solver='liblinear'
)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model trained successfully!")
print(f"📈 Accuracy: {accuracy:.2%}")

# Enhanced prediction function
def predict_emotion_enhanced(text):
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    emotion_probs = {}
    for i, emotion in enumerate(model.classes_):
        emotion_probs[emotion] = probabilities[i]
    
    # Sort by probability
    sorted_probs = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Confidence level
    confidence = emotion_probs[prediction]
    if confidence > 0.6:
        confidence_level = "🟢 HIGH"
    elif confidence > 0.4:
        confidence_level = "🟡 MEDIUM"
    else:
        confidence_level = "🔴 LOW"
    
    print(f"📝 Input: '{text}'")
    print(f"🎯 Predicted Emotion: {prediction.upper()}")
    print(f"📊 Confidence Scores (Top 3):")
    for emotion, prob in sorted_probs:
        print(f"   {emotion}: {prob:.2%}")
    print(f"💪 Confidence Level: {confidence_level}")
    print('-' * 50)
    
    return prediction, emotion_probs

# Test the improved model
print("\n" + "=" * 60)
print("🧪 COMPREHENSIVE MODEL TESTING")
print("=" * 60)

test_samples = [
    "I am extremely happy today!",
    "This situation makes me very angry and frustrated",
    "I feel really sad and lonely",
    "I'm scared about what might happen tomorrow",
    "Wow, this is completely surprising and unexpected!",
    "It's okay, nothing special about this",
    "I love this amazing wonderful thing!",
    "I hate this terrible awful situation"
]

for sample in test_samples:
    predict_emotion_enhanced(sample)

# Interactive testing
print("=" * 60)
print("🎮 INTERACTIVE EMOTION DETECTION")
print("=" * 60)
print("Now you can test any sentence!")
print("Type 'quit' to exit\n")

test_count = 1
while True:
    user_input = input(f"Test #{test_count} - Enter a sentence: ")
    
    if user_input.lower() == 'quit':
        print("👋 Thank you for testing! Goodbye!")
        break
    
    if user_input.strip():
        predict_emotion_enhanced(user_input)
        test_count += 1
    else:
        print("⚠️  Please enter some text")
