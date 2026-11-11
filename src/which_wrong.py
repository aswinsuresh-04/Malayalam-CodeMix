import joblib

model = joblib.load("results/model_v2.pkl")
en_test = ["trailor", "traler", "movie", "movy", "supper", "super", "fight", "fyt",
           "mass", "maas", "hero", "heero", "climax", "climaks", "interval",
           "intervel", "bgm", "dialogue", "dailogue"]
for w in en_test:
    if model.predict([w])[0] != "EN":
        print(w, "->", model.predict([w])[0])