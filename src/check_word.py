import sys
import pandas as pd
from wordfreq import top_n_list, zipf_frequency

w = sys.argv[1].lower()
train = pd.read_csv("data/train_auto.csv", keep_default_na=False)
rows = train[train["word"].str.lower() == w]

print("labels in the training comments:", rows["label"].value_counts().to_dict())
print("English frequency score (3+ = common):", round(zipf_frequency(w, "en"), 2))
print("in the top 30000 English words:", w in top_n_list("en", 30000))