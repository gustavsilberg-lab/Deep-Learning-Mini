import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

SPLITS = ["train", "test", "contrast", "gaussian_noise", "impulse_noise",
          "jpeg_compression", "motion_blur", "pixelate", "spatter"]

def load_gtsrb():
    return load_dataset("parquet", data_files={s: f"data/{s}-00000-of-00001.parquet" for s in SPLITS})

def track_split(labels, val_frac=0.15, seed=42):
    tracks = np.arange(len(labels) // 30)
    tr, va = train_test_split(tracks, test_size=val_frac, stratify=labels[::30], random_state=seed)
    to_idx = lambda t: (t[:, None] * 30 + np.arange(30)).ravel()
    return to_idx(tr), to_idx(va)

def get_transform(size, augment, pretrained):
    mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if pretrained else ([0.5] * 3, [0.5] * 3)
    aug = [T.RandomRotation(10),
           T.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
           T.ColorJitter(brightness=0.3, contrast=0.3)] if augment else []
    return T.Compose([T.Resize((size, size)), *aug, T.ToTensor(), T.Normalize(mean, std)])

class GTSRB(Dataset):
    def __init__(self, split, idx, tf):
        self.split, self.idx, self.tf = split, idx, tf
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        row = self.split[int(self.idx[i])]
        return self.tf(row["image"].convert("RGB")), row["label"]

def get_loaders(ds, size=48, augment=False, pretrained=False, batch_size=64):
    tr_idx, va_idx = track_split(np.array(ds["train"]["label"]))
    train = GTSRB(ds["train"], tr_idx, get_transform(size, augment, pretrained))
    val = GTSRB(ds["train"], va_idx, get_transform(size, False, pretrained))
    return DataLoader(train, batch_size, shuffle=True), DataLoader(val, batch_size)

def get_test_loader(ds, split, size=48, pretrained=False, batch_size=64):
    data = GTSRB(ds[split], np.arange(len(ds[split])), get_transform(size, False, pretrained))
    return DataLoader(data, batch_size)