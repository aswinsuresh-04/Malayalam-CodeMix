import pandas as pd
import re
from nltk.corpus import words

english = set(w.lower() for w in words.words())
english |= {"trailer", "like", "comment", "fans", "super", "superb", "mass", "love", "bgm", "movies", "views"}

ml_fix = {"pole", "undo", "pani", "alle", "ulla", "kola", "ne", "nu", "da", "di", "eda", "edi"}

names = {"manju", "warrier", "mohanlal", "mammootty", "raju", "suraj",
         "sukumaran", "lucifer", "karnataka", "india", "malayalam", "raja",
         "tamil", "tamilnadu", "telugu", "kerala"}

def label_word(w):
    lw = w.lower()
    if len(lw) == 1:
        return "OTHER"
    if lw.endswith(("ettan", "etta", "ikka", "kka")):
        return "ML"
    if lw in names:
        return "NE"
    if lw in ml_fix:
        return "ML"
    if lw in english or (lw.endswith("s") and lw[:-1] in english):
        return "EN"
    return "ML"

df = pd.read_csv("data/roman.csv")

rows = []
for sid, text in enumerate(df["text"]):
    for w in re.findall(r"[A-Za-z]+", str(text)):
        rows.append((sid, w, label_word(w)))

out = pd.DataFrame(rows, columns=["sent_id", "word", "label"])

gold_ids = out["sent_id"].drop_duplicates().sample(50, random_state=0)
out[out.sent_id.isin(gold_ids)].to_csv("data/gold_to_fix.csv", index=False)
out[~out.sent_id.isin(gold_ids)].to_csv("data/train_auto.csv", index=False)
print(out["label"].value_counts())