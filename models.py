import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))

class SmallCNN(nn.Module):
    def __init__(self, n_classes=43, size=48):
        super().__init__()
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * (size // 8) ** 2, 256),
                                        nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, n_classes))
    def forward(self, x):
        return self.classifier(self.features(x))

def build_model(name, size=48, freeze=False):
    if name == "cnn":
        return SmallCNN(size=size)
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if freeze:
        for q in m.parameters():
            q.requires_grad = False
    m.fc = nn.Linear(512, 43)
    return m