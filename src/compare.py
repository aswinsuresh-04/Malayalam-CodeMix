import joblib

old = joblib.load("results/model.pkl")
new = joblib.load("results/model_norm.pkl")

words = ["nannayittund", "nanayitund", "nannayittundu",
         "enthaa", "entha", "enthada",
         "cheyyunnilla", "cheyunilla", "cheyyunilaa",
         "ishtam", "istam", "ishttam",
         "pwoli", "poli", "polii",
         "kandu", "kandoo", "kanduu",
         "ippo", "ipo", "ippoo"]

print(f"{'word':<16}{'old':<8}{'norm':<8}")
for w in words:
    print(f"{w:<16}{old.predict([w])[0]:<8}{new.predict([w])[0]:<8}")

print("\n--- English words with messy spelling (should be EN) ---")
en_words = ["trailor", "traler", "movie", "movy", "supper", "super",
            "fight", "fyt", "mass", "maas", "hero", "heero",
            "climax", "climaks", "interval", "intervel", "bgm", "dialogue", "dailogue"]
for w in en_words:
    print(f"{w:<16}{old.predict([w])[0]:<8}{new.predict([w])[0]:<8}")