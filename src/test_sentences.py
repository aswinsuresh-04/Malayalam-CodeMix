import re
import joblib

old = joblib.load("results/model_noisy.pkl")
new = joblib.load("results/model_v2.pkl")

sentences = [
    "innale paranja movie kandu, scene-okke kidilu aayirunnu",
    "njan ippo office-ilotu irangukaya, 10 minutes-il ethum",
    "njan aake confuse aayi, ningal eethu option aam select cheythathe?",
    "aa shop-il nalla items und, pakshe price korach expensive aane.",
    "phone-il charge kuravayirunnu, athukondaanu call cut aayathe.",
    "Enikku aa proposal kandittu valid aayi thonniyilla.",
    "Njangal nale resort-ilotu poovarund, neeyum koodi varunno?",
    "Food order cheytho, atho namukku purathu poyi kazhikamo?",
    "Enthaayalum aa issue solve aayi, ini no problem.",
    "Avarude performance super aayirunnu, full energy thonni.",
    "Script run cheythal execution time valare high aavunnu, enntaanu reason?",
    "Project deadline missing aavaathirikkaan namukku tasks prioritize cheyyaam.",
    "Database sync cheyyan njan trying aayirunnu, pakshe error message varunnu.",
    "Client meeting rescheduled aayi, new timing mail cheyyaam.",
    "System update cheytha sesham aane eii problem start aayathe.",
    "da Shyam food kazhikan vanna",
]

for s in sentences:
    for t in re.findall(r"[A-Za-z]+", s):
        a, b = old.predict([t])[0], new.predict([t])[0]
        if a != b:
            print(f"{t:<18} old={a:<6} v2={b}")