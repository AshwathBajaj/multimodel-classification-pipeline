import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


# STEP 1: CREATE LABEL CSV

base_path = "lgg-mri-segmentation/kaggle_3m"
data = []

def get_stage(mask):
    tumor_pixels = np.sum(mask > 0)

    if tumor_pixels == 0:
        return 0
    elif tumor_pixels < 500:
        return 1
    elif tumor_pixels < 2000:
        return 2
    else:
        return 3

for patient in os.listdir(base_path):
    patient_path = os.path.join(base_path, patient)

    if not os.path.isdir(patient_path):
        continue

    for file in os.listdir(patient_path):
        if "_mask" in file:
            mask_path = os.path.join(patient_path, file)
            image_path = mask_path.replace("_mask", "")

            mask = cv2.imread(mask_path, 0)
            if mask is None:
                continue

            stage = get_stage(mask)
            data.append([image_path, stage])

df = pd.DataFrame(data, columns=["image_path", "stage"])
df.to_csv("classification_labels.csv", index=False)


# STEP 2: DATASET

class MRIDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["stage"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = MRIDataset(df, transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)


# STEP 3: MODEL (CHANGE HERE TO SWITCH MODELS)

NUM_CLASSES = 4

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

# To switch models:
# model = models.vgg16(pretrained=True)
# model.classifier[6] = nn.Linear(4096, NUM_CLASSES)

# model = models.inception_v3(pretrained=True, aux_logits=False)
# model.fc = nn.Linear(2048, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# STEP 4: LOSS + OPTIMIZER

class OrdinalFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()

        class_indices = torch.arange(logits.size(1)).float().to(logits.device)
        target_indices = targets.unsqueeze(1).float()

        weights = 1 + self.alpha * torch.abs(class_indices - target_indices)
        focal_term = (1 - probs) ** self.gamma

        loss = -weights * focal_term * targets_one_hot * torch.log(probs + 1e-8)
        return loss.sum(dim=1).mean()

criterion = OrdinalFocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# STEP 5: TRAINING LOOP

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

print("Training Complete!")