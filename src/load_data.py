from datasets import load_dataset
import pandas as pd

ds = load_dataset("community-datasets/offenseval_dravidian", "malayalam")
df = pd.DataFrame(ds["train"])
print(df.columns)
for s in df["text"].head(20):
    print(s)
df.to_csv("data/train.csv", index=False)