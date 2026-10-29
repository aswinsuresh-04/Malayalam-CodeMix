import pandas as pd

g = pd.read_csv("data/gold_fixed.csv")
print(g["label"].value_counts())
bad = g[~g["label"].isin(["ML", "EN", "NE", "OTHER"])]
print("Bad labels:", len(bad))
print(bad)