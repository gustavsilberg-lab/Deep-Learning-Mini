import os
from data import load_gtsrb

ds = load_gtsrb()["test"]
os.makedirs("samples", exist_ok=True)
n = {}
for i, y in enumerate(ds["label"]):
    if n.get(y, 0) < 2:
        ds[i]["image"].save(f"samples/class{y:02d}_{n.get(y, 0)}.png"); n[y] = n.get(y, 0) + 1