import argparse, json, os
import numpy as np, torch, matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from data import SPLITS, load_gtsrb, get_test_loader, get_transform
from models import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--split", default="all")
ap.add_argument("--samples")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(args.checkpoint, map_location=device)
a = ckpt["args"]
model = build_model(a["model"], a["img_size"]).to(device)
model.load_state_dict(ckpt["model"]); model.eval()
pretrained = a["model"] == "resnet18"

if args.samples:
    tf = get_transform(a["img_size"], False, pretrained)
    with torch.no_grad():
        for f in sorted(os.listdir(args.samples)):
            x = tf(Image.open(os.path.join(args.samples, f)).convert("RGB"))[None].to(device)
            print(f, "-> predicted class", model(x).argmax(1).item())
    raise SystemExit

ds = load_gtsrb()
name = os.path.splitext(os.path.basename(args.checkpoint))[0]
results = {}
for split in (SPLITS[1:] if args.split == "all" else [args.split]):
    preds, ys = [], []
    with torch.no_grad():
        for x, y in get_test_loader(ds, split, a["img_size"], pretrained):
            preds.append(model(x.to(device)).argmax(1).cpu()); ys.append(y)
    preds, ys = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    cm = confusion_matrix(ys, preds, labels=range(43))
    results[split] = {"accuracy": float((preds == ys).mean()),
                      "macro_f1": float(f1_score(ys, preds, average="macro")),
                      "per_class_acc": (cm.diagonal() / cm.sum(1)).tolist()}
    print(f"{split}: acc {results[split]['accuracy']:.4f} | macro F1 {results[split]['macro_f1']:.4f}")
    if split == "test":
        fig, ax = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay(cm).plot(ax=ax, include_values=False, cmap="Blues", xticks_rotation="vertical")
        ax.tick_params(labelsize=6)
        plt.title(name); plt.savefig(f"figures/cm_{name}.png", dpi=200, bbox_inches="tight"); plt.close()

json.dump(results, open(f"results/{name}_test.json", "w"), indent=1)