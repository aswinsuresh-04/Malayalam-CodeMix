import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

train = pd.read_csv("data/train_auto.csv", keep_default_na=False)
gold = pd.read_csv("data/gold_fixed.csv", keep_default_na=False)

train["word"] = train["word"].str.lower()
gold["word"] = gold["word"].str.lower()

model = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4)),
    LogisticRegression(max_iter=1000),
)
model.fit(train["word"], train["label"])

import joblib
joblib.dump(model, "results/model.pkl")

pred = model.predict(gold["word"])
print("Accuracy:", round(accuracy_score(gold["label"], pred), 3))
print(classification_report(gold["label"], pred, zero_division=0))

gold["pred"] = pred
errors = gold[gold["label"] != gold["pred"]]
print("\nMistakes:")
print(errors[["word", "label", "pred"]].head(30))