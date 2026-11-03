import re
import joblib

model = joblib.load("results/model.pkl")

NAMES = {"ML": "Malayalam", "EN": "English", "NE": "Name", "OTHER": "Other"}

def identify(sentence):
    words = re.findall(r"[A-Za-z]+", sentence)
    if not words:
        return []
    labels = model.predict([w.lower() for w in words])
    return list(zip(words, labels))

print("Type a code-mixed sentence (or 'q' to quit)\n")
while True:
    s = input("> ")
    if s.strip().lower() == "q":
        break
    for word, label in identify(s):
        print(f"  {word:<20} {NAMES[label]}")
    print()