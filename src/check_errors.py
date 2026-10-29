import pandas as pd

df = pd.read_csv("data/labeled_auto.csv")

print("=== Most common words labeled EN (look for Malayalam ones) ===")
print(df[df.label == "EN"]["word"].str.lower().value_counts().head(60))

print("\n=== Most common words labeled ML (look for English ones) ===")
print(df[df.label == "ML"]["word"].str.lower().value_counts().head(60))