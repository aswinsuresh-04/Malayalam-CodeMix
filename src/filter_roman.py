import pandas as pd
import re

df = pd.read_csv("data/train.csv")

def is_roman(text):
    text = str(text)
    ml = len(re.findall(r'[\u0D00-\u0D7F]', text))   # Malayalam script letters
    en = len(re.findall(r'[A-Za-z]', text))          # English letters
    return en > 0 and ml == 0

roman = df[df["text"].apply(is_roman)]
print("Total:", len(df))
print("Roman only:", len(roman))

roman[["text"]].to_csv("data/roman.csv", index=False)

for s in roman["text"].sample(20, random_state=1):
    print(s)