import argparse, csv, os, time
import numpy as np, torch, torch.nn as nn
from data import load_gtsrb, get_loaders
from models import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
ap.add_argument("--img-size", type=int, default=48)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--patience", type=int, default=5)
ap.add_argument("--augment", action="store_true")
ap.add_argument("--weighted-loss", action="store_true")
ap.add_argument("--freeze-backbone", action="store_true")
ap.add_argument("--run-name", required=True)
args = ap.parse_args()

torch.manual_seed(42); np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
ds = load_gtsrb()
train_dl, val_dl = get_loaders(ds, args.img_size, args.augment, args.model == "resnet18", args.batch_size)
model = build_model(args.model, args.img_size, args.freeze_backbone).to(device)
print("Device:", device, "| Parameters:", sum(q.numel() for q in model.parameters()))

weights = None
if args.weighted_loss:
    counts = np.bincount(np.array(ds["train"]["label"]), minlength=43)
    weights = torch.tensor(counts.sum() / (43 * counts), dtype=torch.float32, device=device)
loss_fn = nn.CrossEntropyLoss(weight=weights)
opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

def run_epoch(dl, train):
    model.train(train and not args.freeze_backbone)
    total_loss, correct, n = 0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return total_loss / n, correct / n

os.makedirs("checkpoints", exist_ok=True); os.makedirs("results", exist_ok=True)
best, bad, history = 0, 0, []
for epoch in range(1, args.epochs + 1):
    t = time.time()
    tr_loss, tr_acc = run_epoch(train_dl, True)
    va_loss, va_acc = run_epoch(val_dl, False)
    sched.step()
    history.append([epoch, tr_loss, tr_acc, va_loss, va_acc, time.time() - t])
    print(f"Epoch {epoch}: train {tr_loss:.3f}/{tr_acc:.3f} | val {va_loss:.3f}/{va_acc:.3f} | {time.time() - t:.0f}s")
    if va_acc > best:
        best, bad = va_acc, 0
        torch.save({"model": model.state_dict(), "args": vars(args)}, f"checkpoints/{args.run_name}.pt")
    else:
        bad += 1
        if bad >= args.patience:
            print("Early stopping"); break

with open(f"results/{args.run_name}_history.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"]); w.writerows(history)
print("Best val acc:", best)