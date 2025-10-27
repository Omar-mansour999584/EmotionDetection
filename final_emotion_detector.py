# Final Emotion Detection - Enhanced Version
print("🎯 FINAL EMOTION DETECTION PROJECT")
print("=" * 50)

# Enhanced training data with more examples
texts = [
    # Happy (8 examples)
    "I am happy and joyful", "I feel great today", "This is wonderful news",
    "I love this so much", "I'm excited and thrilled", "This makes me happy",
    "Feeling fantastic today", "I'm overjoyed with happiness",
    
    # Sad (8 examples) 
    "I feel sad and depressed", "This is terrible news", "I'm feeling down",
    "This makes me unhappy", "I feel miserable today", "This is so sad",
    "I'm heartbroken", "Feeling very depressed",
    
    # Angry (8 examples)
    "I am angry and mad", "This makes me furious", "I hate this situation",
    "I'm really pissed off", "This is so annoying", "I feel outraged",
    "This irritates me", "I'm boiling with anger",
    
    # Fear (8 examples)
    "I feel scared and afraid", "This is frightening", "I'm terrified",
    "This makes me anxious", "I'm worried about this", "I feel threatened",
    "This is scary", "I'm panicking right now",
    
    # Surprise (8 examples)
    "I am surprised and amazed", "Wow this is incredible", "This is shocking",
    "I'm stunned by this news", "What a surprise", "This is unbelievable",
    "I'm astonished", "This caught me off guard",
    
    # Neutral (8 examples)
    "I feel neutral and okay", "This is normal", "Nothing special",
    "I'm indifferent about this", "This is average", "It's fine",
    "This is ordinary", "I don't feel strongly"
]

emotions = [
    'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy',
    'sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad',
    'angry', 'angry', 'angry', 'angry', 'angry', 'angry', 'angry', 'angry', 
    'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear',
    'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise',
    'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral'
]

print(f"📊 Training with {len(texts)} samples")
print(f"🎯 Emotions: {set(emotions)}")

# Train model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)

# Split data for testing
X_train, X_test, y_train, y_test = train_test_split(X, emotions, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"📈 Accuracy: {accuracy:.2%}")

# Enhanced prediction function
def predict_emotion(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    # Get top 3 emotions with confidence
    emotion_probs = {}
    for i, emotion in enumerate(model.classes_):
        emotion_probs[emotion] = probabilities[i]
    
    # Sort by probability
    sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📝 Input: '{text}'")
    print(f"🎯 Predicted Emotion: {prediction.upper()}")
    print("📊 Confidence Scores (Top 3):")
    
    for emotion, prob in sorted_emotions[:3]:
        print(f"   {emotion}: {prob:.2%}")
    
    # Confidence level
    top_prob = sorted_emotions[0][1]
    if top_prob > 0.7:
        confidence = "🟢 HIGH"
    elif top_prob > 0.5:
        confidence = "🟡 MEDIUM"
    else:
        confidence = "🔴 LOW"
    
    print(f"💪 Confidence Level: {confidence}")
    
    return prediction

# Comprehensive testing
print("\n" + "=" * 60)
print("🧪 COMPREHENSIVE MODEL TESTING")
print("=" * 60)

test_cases = [
    "I am extremely happy today!",
    "This situation makes me very angry and frustrated",
    "I feel really sad and lonely",
    "I'm scared about what might happen tomorrow", 
    "Wow, this is completely surprising and unexpected!",
    "It's okay, nothing special about this",
    "I love this amazing wonderful thing!",
    "I hate this terrible awful situation"
]

for test in test_cases:
    predict_emotion(test)
    print("-" * 50)

# Interactive session
print("\n" + "=" * 60)
print("🎮 INTERACTIVE EMOTION DETECTION")
print("=" * 60)
print("Now you can test any sentence!")
print("Type 'quit' to exit\n")

test_count = 0
while True:
    user_input = input(f"Test #{test_count + 1} - Enter a sentence: ").strip()
    
    if user_input.lower() == 'quit':
        break
    elif user_input:
        predict_emotion(user_input)
        test_count += 1
        print("=" * 50)
    else:
        print("Please enter some text!")

# Final summary
print("\n" + "=" * 60)
print("📊 PROJECT SUMMARY")
print("=" * 60)
print(f"✅ Total training samples: {len(texts)}")
print(f"✅ Model accuracy: {accuracy:.2%}")
print(f"✅ Emotions detected: {set(emotions)}")
print(f"✅ Interactive tests performed: {test_count}")
print("\n🎉 EMOTION DETECTION PROJECT COMPLETED SUCCESSFULLY!")
print("🚀 Your AI model is ready to detect emotions from text!")
print("=" * 60)
