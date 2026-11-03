import re
import joblib

MODEL_PATH = "results/model_v2.pkl"
model = joblib.load(MODEL_PATH)

NAMES = {"ML": "Malayalam", "EN": "English", "NE": "Name", "OTHER": "Other"}
ENGLISH_LETTERS = {"a", "i"}      # the only English words that are one letter

def identify(sentence):
    tokens = re.findall(r"[\u0D00-\u0D7F]+|[A-Za-z]+|\d+", sentence)
    result = []
    for t in tokens:
        if re.match(r"[\u0D00-\u0D7F]", t):
            result.append((t, "ML"))              # Malayalam script is always Malayalam
        elif t.isdigit():
            result.append((t, "OTHER"))           # numbers
        elif len(t) == 1:
            result.append((t, "EN" if t.lower() in ENGLISH_LETTERS else "OTHER"))
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