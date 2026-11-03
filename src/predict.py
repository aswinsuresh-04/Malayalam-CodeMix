import re
import joblib

MODEL_PATH = "results/model_norm.pkl"        # change to results/model_norm.pkl if Step 2 was better
model = joblib.load(MODEL_PATH)

NAMES = {"ML": "Malayalam", "EN": "English", "NE": "Name", "OTHER": "Other"}

def identify(sentence):
    # words in Malayalam script OR English letters; punctuation, numbers, emojis are skipped
    tokens = re.findall(r"[\u0D00-\u0D7F]+|[A-Za-z]+", sentence)
    result = []
    for t in tokens:
        if re.match(r"[\u0D00-\u0D7F]", t):
            result.append((t, "ML"))                 # Malayalam script is always Malayalam
        else:
            result.append((t, model.predict([t])[0]))
    return result

print("Type a code-mixed sentence (or 'q' to quit)\n")
while True:
    s = input("> ")
    if s.strip().lower() == "q":
        break
    for word, label in identify(s):
        print(f"  {word:<20} {NAMES[label]}")
    print()