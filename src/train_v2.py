import random
import pandas as pd
import joblib
from wordfreq import top_n_list
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import classification_report, accuracy_score
from features import norm

random.seed(0)
COMMON_N = 6000      # how many common English words to add (you can try 3000 or 10000)

def make_typo(w):
    w = w.lower()
    vowels = [i for i, c in enumerate(w) if c in "aeiou"]
    r = random.random()
    if r < 0.35 and vowels:
        i = random.choice(vowels)
        return w[:i] + w[i] + w[i:]
    if r < 0.55 and vowels:
        i = random.choice(vowels)
        return w[:i] + w[i + 1:]
    if r < 0.75:
        i = random.randrange(len(w) - 1)
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return w.replace("c", "k").replace("ph", "f")

train = pd.read_csv("data/train_auto.csv", keep_default_na=False)
gold = pd.read_csv("data/gold_fixed.csv", keep_default_na=False)

# words that show up in the comments as Malayalam/names: never add these as English
lc = train["word"].str.lower()
vc = lc[train["label"].isin(["ML", "NE"])].value_counts()
seen_ml = set(vc[vc >= 3].index)

# most common English words (by frequency), not random dictionary words
common = []
for w in top_n_list("en", 30000):
    if w.isascii() and w.isalpha() and len(w) >= 4 and w not in seen_ml:
        common.append(w)
    if len(common) >= COMMON_N:
        break

en_words = list(lc[train["label"] == "EN"].unique()) + common
en_words = [w for w in en_words if len(w) >= 4]

noisy = []
for w in en_words:
    t = make_typo(w)
    if t != w and len(t) >= 3:
        noisy.append((t, "EN"))
noisy_df = pd.DataFrame(noisy, columns=["word", "label"])
common_df = pd.DataFrame({"word": common, "label": "EN"})
print("Common English words added:", len(common_df), "| noisy copies:", len(noisy_df))

data = pd.concat([train[["word", "label"]], noisy_df, common_df])

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

ml_test = ["nannayittund", "nanayitund", "nannayittundu", "enthaa", "entha", "enthada",
           "cheyyunnilla", "cheyunilla", "cheyyunilaa", "ishtam", "istam", "ishttam",
           "pwoli", "poli", "polii", "kandu", "kandoo", "kanduu", "ippo", "ipo", "ippoo"]
en_test = ["trailor", "traler", "movie", "movy", "supper", "super", "fight", "fyt",
           "mass", "maas", "hero", "heero", "climax", "climaks", "interval",
           "intervel", "bgm", "dialogue", "dailogue"]
print("Malayalam variants correct:", sum(model.predict([w])[0] == "ML" for w in ml_test), "/", len(ml_test))
print("English variants correct:  ", sum(model.predict([w])[0] == "EN" for w in en_test), "/", len(en_test))

# words that failed in your sentence test
fail_ml = ["aa", "aam", "da"]
fail_en = ["valid", "deadline", "prioritize", "database", "rescheduled"]
print("Earlier failures fixed (Malayalam):", sum(model.predict([w])[0] == "ML" for w in fail_ml), "/", len(fail_ml))
print("Earlier failures fixed (English):  ", sum(model.predict([w])[0] == "EN" for w in fail_en), "/", len(fail_en))

joblib.dump(model, "results/model_v2.pkl")