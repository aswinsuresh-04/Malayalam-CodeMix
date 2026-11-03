import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import classification_report, accuracy_score
from features import norm

train = pd.read_csv("data/train_auto.csv", keep_default_na=False)
gold = pd.read_csv("data/gold_fixed.csv", keep_default_na=False)

# model sees the original spelling AND the cleaned spelling
model = make_pipeline(
    make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), preprocessor=str.lower),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), preprocessor=norm),
    ),
    LogisticRegression(max_iter=1000),
)
model.fit(train["word"], train["label"])

pred = model.predict(gold["word"])
print("Accuracy:", round(accuracy_score(gold["label"], pred), 3))
print(classification_report(gold["label"], pred, zero_division=0))

joblib.dump(model, "results/model_norm.pkl")