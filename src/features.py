import re

def norm(w):
    """Make spelling variants look alike: ippo/ipo, cheyyunnilla/cheyunilla"""
    w = w.lower()
    w = re.sub(r"(.)\1+", r"\1", w)          # double letters -> single
    w = w.replace("th", "t").replace("dh", "d").replace("sh", "s")
    return w