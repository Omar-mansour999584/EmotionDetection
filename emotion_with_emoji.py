# Emotion Detection with Comprehensive Emoji Support
print("🎭 Emotion Detection with Complete Emoji Support")
print("=" * 60)

# Enhanced training data with COMPREHENSIVE EMOJIS
texts = []
emotions = []

# Happy emotions with extensive emojis
happy_texts = [
    "I am happy and joyful 😊", "I feel great today 😄", "This is wonderful news 🎉",
    "I love this so much ❤️", "I'm excited and thrilled 🤩", "This makes me happy 😁",
    "Feeling fantastic today ✨", "I'm overjoyed with happiness 🌟", "So happy right now 💃",
    "This is amazing 🥳", "I'm delighted 😃", "Feeling blessed 🙏", "Wonderful time 🎊",
    "I'm ecstatic 🥰", "This is perfect 💫", "So much fun 🎈", "I'm grinning 😸",
    "Absolutely wonderful 🌈", "My heart is full 💗", "Best day ever 🏆"
]
texts.extend(happy_texts)
emotions.extend(['happy'] * len(happy_texts))

# Sad emotions with extensive emojis
sad_texts = [
    "I feel sad and depressed 😢", "This is terrible news 😞", "I'm feeling down 😔",
    "This makes me unhappy 😕", "I feel miserable today 💔", "This is so sad 😭",
    "I'm heartbroken 🥺", "Feeling very depressed 🌧️", "So disappointed 😣",
    "This is heartbreaking 💔", "I'm devastated 😫", "Feeling lonely 🏚️",
    "Tears in my eyes 🥲", "This hurts 😿", "I'm so upset 😥", "Feeling blue 💙",
    "My heart is heavy ⚖️", "This is tragic 🕯️", "I'm grieving 😔", "So sorrowful 🍂"
]
texts.extend(sad_texts)
emotions.extend(['sad'] * len(sad_texts))

# Angry emotions with extensive emojis
angry_texts = [
    "I am angry and mad 😠", "This makes me furious 💢", "I hate this situation 👿",
    "I'm really pissed off 🤬", "This is so annoying 😤", "I feel outraged 😡",
    "This irritates me 🙄", "I'm boiling with anger 🔥", "So frustrated 😾",
    "This infuriates me 💥", "I'm livid 🗯️", "Absolutely furious 🌋",
    "This makes me see red 🚩", "I'm fuming 💨", "So angry right now ⚡",
    "This is unacceptable 🚫", "I'm enraged 😈", "Blood is boiling 🩸",
    "This is irritating 💣", "I'm seething 🐍"
]
texts.extend(angry_texts)
emotions.extend(['angry'] * len(angry_texts))

# Fear emotions with extensive emojis
fear_texts = [
    "I feel scared and afraid 😨", "This is frightening 😰", "I'm terrified 😱",
    "This makes me anxious 😟", "I'm worried about this 😓", "I feel threatened 🚨",
    "This is scary 👻", "I'm panicking right now 💦", "So nervous 🥶",
    "This is terrifying 🕷️", "I'm apprehensive 😥", "Feeling uneasy 🕳️",
    "This gives me chills 🧊", "I'm frightened 🦇", "So anxious right now 📉",
    "This is alarming ⚠️", "I'm petrified 🗿", "Heart is racing 💓",
    "This is daunting 🏔️", "I'm intimidated 🦁"
]
texts.extend(fear_texts)
emotions.extend(['fear'] * len(fear_texts))

# Surprise emotions with extensive emojis
surprise_texts = [
    "I am surprised and amazed 😲", "Wow this is incredible 🤯", "This is shocking 😮",
    "I'm stunned by this news 🎊", "What a surprise 🎁", "This is unbelievable 😳",
    "I'm astonished ✨", "This caught me off guard ⚡", "Absolutely amazed 🌠",
    "This is mind-blowing 🧠", "I'm speechless 🤐", "What a shock 💫",
    "This is astounding 🌌", "I'm flabbergasted 🫢", "Completely surprised 🎪",
    "This is incredible 🦄", "I'm bewildered 🌀", "Totally unexpected 🎲",
    "This is phenomenal 🌟", "I'm dumbfounded 🫣"
]
texts.extend(surprise_texts)
emotions.extend(['surprise'] * len(surprise_texts))

# Neutral emotions with extensive emojis
neutral_texts = [
    "I feel neutral and okay 😐", "This is normal 👌", "Nothing special 🤷",
    "I'm indifferent about this 💤", "This is average 📊", "It's fine ✅",
    "This is ordinary 📝", "I don't feel strongly 🎯", "Meh 🤨",
    "This is whatever 🎭", "I'm impartial ⚖️", "Not good not bad 🔄",
    "This is standard 🏷️", "I'm neutral 🫥", "Nothing remarkable 📍",
    "This is mediocre 🎪", "I'm unimpressed 😑", "So-so situation 🔀",
    "This is passable 📋", "I'm nonchalant 🍃"
]
texts.extend(neutral_texts)
emotions.extend(['neutral'] * len(neutral_texts))

print(f"📊 Training with {len(texts)} samples")
print(f"🎯 Emotions: {set(emotions)}")

# Train model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(max_features=1500)
X = vectorizer.fit_transform(texts)

model = LogisticRegression(random_state=42, max_iter=2000)
model.fit(X, emotions)

print("✅ Model trained with comprehensive emoji support!")

# Comprehensive emoji mapping
emoji_to_emotion = {
    # Happy emojis
    '😊': 'happy', '😄': 'happy', '😃': 'happy', '😁': 'happy', '🤩': 'happy',
    '🥰': 'happy', '😍': 'happy', '🤗': 'happy', '😎': 'happy', '🥳': 'happy',
    '😸': 'happy', '😻': 'happy', '💃': 'happy', '🕺': 'happy', '🎉': 'happy',
    '🎊': 'happy', '✨': 'happy', '🌟': 'happy', '❤️': 'happy', '💖': 'happy',
    '💕': 'happy', '💗': 'happy', '💓': 'happy', '🙌': 'happy', '👏': 'happy',
    '👍': 'happy', '🎈': 'happy', '🏆': 'happy', '🌈': 'happy', '⭐': 'happy',
    
    # Sad emojis  
    '😢': 'sad', '😞': 'sad', '😔': 'sad', '😕': 'sad', '😭': 'sad',
    '🥺': 'sad', '😣': 'sad', '😫': 'sad', '😥': 'sad', '😓': 'sad',
    '😿': 'sad', '💔': 'sad', '🌧️': 'sad', '🥲': 'sad', '🏚️': 'sad',
    '💙': 'sad', '⚖️': 'sad', '🕯️': 'sad', '🍂': 'sad', '❄️': 'sad',
    
    # Angry emojis
    '😠': 'angry', '😡': 'angry', '🤬': 'angry', '😤': 'angry', '💢': 'angry',
    '👿': 'angry', '😾': 'angry', '💥': 'angry', '🗯️': 'angry', '🌋': 'angry',
    '🚩': 'angry', '💨': 'angry', '⚡': 'angry', '🚫': 'angry', '😈': 'angry',
    '🩸': 'angry', '💣': 'angry', '🐍': 'angry', '🔥': 'angry', '💀': 'angry',
    
    # Fear emojis
    '😨': 'fear', '😰': 'fear', '😱': 'fear', '😟': 'fear', '😓': 'fear',
    '🥶': 'fear', '😥': 'fear', '🚨': 'fear', '👻': 'fear', '💦': 'fear',
    '🕷️': 'fear', '🕳️': 'fear', '🧊': 'fear', '🦇': 'fear', '📉': 'fear',
    '⚠️': 'fear', '🗿': 'fear', '💓': 'fear', '🏔️': 'fear', '🦁': 'fear',
    
    # Surprise emojis
    '😲': 'surprise', '🤯': 'surprise', '😮': 'surprise', '😳': 'surprise',
    '🫢': 'surprise', '🫣': 'surprise', '🎊': 'surprise', '🎁': 'surprise',
    '✨': 'surprise', '⚡': 'surprise', '🌠': 'surprise', '🧠': 'surprise',
    '🤐': 'surprise', '💫': 'surprise', '🌌': 'surprise', '🎪': 'surprise',
    '🦄': 'surprise', '🌀': 'surprise', '🎲': 'surprise', '🌟': 'surprise',
    
    # Neutral emojis
    '😐': 'neutral', '🫥': 'neutral', '😑': 'neutral', '🤨': 'neutral',
    '🙄': 'neutral', '💤': 'neutral', '📊': 'neutral', '✅': 'neutral',
    '📝': 'neutral', '🎯': 'neutral', '🤷': 'neutral', '🎭': 'neutral',
    '⚖️': 'neutral', '🔄': 'neutral', '🏷️': 'neutral', '📍': 'neutral',
    '🎪': 'neutral', '🔀': 'neutral', '📋': 'neutral', '🍃': 'neutral'
}

def predict_emotion_comprehensive(text):
    # Handle pure emoji input
    if text.strip() in emoji_to_emotion:
        predicted_emotion = emoji_to_emotion[text.strip()]
        print(f"\n📝 Input: '{text}'")
        print(f"🎯 Predicted Emotion: {predicted_emotion.upper()}")
        print(f"💡 Detection: Direct emoji mapping")
        print(f"🔤 Emoji Category: {get_emoji_category(text)}")
        return predicted_emotion
    
    # Handle text with emojis
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    emotion_probs = {}
    for i, emotion in enumerate(model.classes_):
        emotion_probs[emotion] = probabilities[i]
    
    sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📝 Input: '{text}'")
    print(f"🎯 Predicted Emotion: {prediction.upper()}")
    print("📊 Confidence Scores (Top 3):")
    
    for emotion, prob in sorted_emotions[:3]:
        print(f"   {emotion}: {prob:.2%}")
    
    top_prob = sorted_emotions[0][1]
    if top_prob > 0.8:
        confidence = "🟢 HIGH"
    elif top_prob > 0.6:
        confidence = "🟡 MEDIUM"
    else:
        confidence = "🔴 LOW"
    
    print(f"💪 Confidence Level: {confidence}")
    
    # Show emoji analysis if present
    emojis_in_text = [char for char in text if char in emoji_to_emotion]
    if emojis_in_text:
        print(f"🔍 Emojis detected: {', '.join(emojis_in_text)}")
        for emoji in emojis_in_text:
            print(f"   {emoji} → {emoji_to_emotion[emoji]}")
    
    return prediction

def get_emoji_category(emoji):
    categories = {
        'happy': ['😊', '😄', '😃', '😁', '🤩', '🥰', '😍', '🤗', '😎', '🥳'],
        'sad': ['😢', '😞', '😔', '😕', '😭', '🥺', '😣', '😫', '😥', '😓'],
        'angry': ['😠', '😡', '🤬', '😤', '💢', '👿', '😾', '💥', '🗯️', '🌋'],
        'fear': ['😨', '😰', '😱', '😟', '😓', '🥶', '😥', '🚨', '👻', '💦'],
        'surprise': ['😲', '🤯', '😮', '😳', '🫢', '🫣', '🎊', '🎁', '✨', '⚡'],
        'neutral': ['😐', '🫥', '😑', '🤨', '🙄', '💤', '📊', '✅', '📝', '🎯']
    }
    
    for category, emoji_list in categories.items():
        if emoji in emoji_list:
            return category
    return "unknown"

# Comprehensive testing
print("\n" + "=" * 70)
print("🧪 COMPREHENSIVE EMOJI TESTING")
print("=" * 70)

# Test popular emojis
popular_emojis = [
    '😂', '❤️', '🤣', '👍', '😭', '🙏', '😘', '🥰', '😍', '😊',
    '🎉', '🤔', '😎', '🌹', '🔥', '💕', '😁', '✨', '🎂', '💖',
    '😢', '😡', '😱', '😴', '🥺', '🤩', '🙄', '💀', '👀', '💯'
]

print("Testing 30+ popular emojis:")
for i, emoji in enumerate(popular_emojis, 1):
    print(f"\n{i:2d}. ", end="")
    predict_emotion_comprehensive(emoji)
    print("-" * 50)

# Test mixed content
print("\n" + "=" * 70)
print("🧪 TESTING MIXED TEXT AND EMOJIS")
print("=" * 70)

mixed_tests = [
    "I'm so happy! 😊🎉❤️",
    "This makes me angry 😡🤬💢",
    "I feel sad today 😢💔🌧️", 
    "I'm scared 😨😱🚨",
    "Wow! Surprising! 😲🤯🎁",
    "It's okay 😐🤷💤",
    "I love you! ❤️🥰💕",
    "This is terrible 😞😭💔",
    "Amazing news! 🤩🎊✨",
    "I'm tired 😴💤🛌"
]

for test in mixed_tests:
    predict_emotion_comprehensive(test)
    print("=" * 60)

# Interactive session
print("\n" + "=" * 70)
print("🎮 INTERACTIVE TESTING - COMPREHENSIVE EMOJI SUPPORT")
print("=" * 70)
print("Now you can test any emoji or text!")
print("Type 'quit' to exit\n")

emoji_count = 0
text_count = 0

while True:
    user_input = input("Enter text/emoji: ").strip()
    
    if user_input.lower() == 'quit':
        break
    elif user_input:
        if any(char in emoji_to_emotion for char in user_input):
            emoji_count += 1
        else:
            text_count += 1
            
        predict_emotion_comprehensive(user_input)
        print(f"📈 Stats: Emojis: {emoji_count}, Text: {text_count}")
        print("=" * 60)
    else:
        print("Please enter some text or emoji!")

print(f"\n🎉 Final Statistics:")
print(f"   Emoji tests: {emoji_count}")
print(f"   Text tests: {text_count}")
print(f"   Total tests: {emoji_count + text_count}")
print("\n🚀 Emotion Detection with Comprehensive Emoji Support Completed!")
