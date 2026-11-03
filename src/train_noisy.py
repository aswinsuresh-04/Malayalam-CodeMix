import random
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import classification_report, accuracy_score
from features import norm

random.seed(0)

def make_typo(w):
    w = w.lower()
    vowels = [i for i, c in enumerate(w) if c in "aeiou"]
    r = random.random()
    if r < 0.35 and vowels:                  # double a vowel: hero -> heero
        i = random.choice(vowels)
        return w[:i] + w[i] + w[i:]
    if r < 0.55 and vowels:                  # drop a vowel: movie -> movi
        i = random.choice(vowels)
        return w[:i] + w[i + 1:]
    if r < 0.75:                             # swap two neighbouring letters
        i = random.randrange(len(w) - 1)
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return w.replace("c", "k").replace("ph", "f")   # sound-based spelling

train = pd.read_csv("data/train_auto.csv", keep_default_na=False)
gold = pd.read_csv("data/gold_fixed.csv", keep_default_na=False)

# noisy English copies
en_words = train[train["label"] == "EN"]["word"].str.lower().unique()
en_words = [w for w in en_words if len(w) >= 4]
noisy = []
for w in en_words:
    t = make_typo(w)
    if t != w and len(t) >= 3:
        noisy.append((t, "EN"))
noisy_df = pd.DataFrame(noisy, columns=["word", "label"])
print("Noisy English words added:", len(noisy_df))

data = pd.concat([train[["word", "label"]], noisy_df])

model = make_pipeline(
    make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), preprocessor=str.lower),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), preprocessor=norm),
    ),
    LogisticRegression(max_iter=1000),
)
model.fit(data["word"], data["label"])

pred = model.predict(gold["word"])
print("Accuracy:", round(accuracy_score(gold["label"], pred), 3))
print(classification_report(gold["label"], pred, zero_division=0))

# spelling-variant checks (same words as compare.py)
ml_test = ["nannayittund", "nanayitund", "nannayittundu", "enthaa", "entha", "enthada",
           "cheyyunnilla", "cheyunilla", "cheyyunilaa", "ishtam", "istam", "ishttam",
           "pwoli", "poli", "polii", "kandu", "kandoo", "kanduu", "ippo", "ipo", "ippoo"]
en_test = ["trailor", "traler", "movie", "movy", "supper", "super", "fight", "fyt",
           "mass", "maas", "hero", "heero", "climax", "climaks", "interval",
           "intervel", "bgm", "dialogue", "dailogue"]
print("Malayalam variants correct:", sum(model.predict([w])[0] == "ML" for w in ml_test), "/", len(ml_test))
print("English variants correct:  ", sum(model.predict([w])[0] == "EN" for w in en_test), "/", len(en_test))

joblib.dump(model, "results/model_noisy.pkl")