import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

# Function to create dataframe from the COVID and Normal directories
def create_df(covid_path, normal_path):
    dd = {"images": [], "labels": []}

    # Add COVID images
    for img in os.listdir(covid_path):
        dd["images"].append(os.path.join(covid_path, img))
        dd["labels"].append("COVID")

    # Add Normal images
    for img in os.listdir(normal_path):
        dd["images"].append(os.path.join(normal_path, img))
        dd["labels"].append("Normal")

    return pd.DataFrame(dd)

# Set the paths to the dataset
covid_path = r"C:\Users\DELL\Desktop\covid19\Covid"
normal_path = r"C:\Users\DELL\Desktop\covid19\Normal"

# Create dataframes
df = create_df(covid_path, normal_path)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle dataset

# Encode labels
le = LabelEncoder()
df["labels"] = le.fit_transform(df["labels"].values)

# Hyperparameters
EPOCHS = 20
LR = 0.1
GAMMA = 0.1
STEP = 10
BATCH = 32
IMG_SIZE = 224
OUT_SIZE = 2  # For binary classification: COVID or Normal

# Split data into training and validation sets
train, val = train_test_split(df.values, random_state=42, test_size=0.2)

# Data pipeline
class Pipeline(Dataset):
    def __init__(self, data, transform):
        super(Pipeline, self).__init__()
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, x):
        img, label = self.data[x, 0], self.data[x, 1]
        img = Image.open(img).convert("RGB")
        img = np.array(img)
        img = self.transform(img)
        return img, label

# Data transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = Pipeline(train, transform)
val_ds = Pipeline(val, transform)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

# Defining ResNet50 model
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, OUT_SIZE)  # Change output to 2 classes: COVID and Normal

# Model class with softmax
class COVID_Detector(nn.Module):
    def __init__(self, model):
        super(COVID_Detector, self).__init__()
        self.model = model
        
    def forward(self, x):
        return nn.functional.softmax(self.model(x), dim=1)

# Training device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Model setup
model = COVID_Detector(resnet)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)

# Training loop
best_model = deepcopy(model)
best_acc = 0

train_loss = []
train_acc = []
val_loss = []
val_acc = []

for i in range(1, EPOCHS+1):
    model.train()
    
    diff = 0
    acc = 0
    total = 0
    
    for data, target in train_dl:
        optimizer.zero_grad()
        
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
            
        out = model(data)
        loss = criterion(out, target)
        diff += loss.item()
        acc += (out.argmax(1) == target).sum().item()
        total += out.size(0)
        loss.backward()
        optimizer.step()
    train_loss += [diff/total]
    train_acc += [acc/total]
    
    # Validation
    model.eval()
    diff = 0
    acc = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_dl:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            out = model(data)
            loss = criterion(out, target)
            diff += loss.item()
            acc += (out.argmax(1) == target).sum().item()
            total += out.size(0)

    val_loss += [diff/total]
    val_acc += [acc/total]
    
    if best_acc < val_acc[-1]:
        best_acc = val_acc[-1]
        best_model = deepcopy(model)
    
    print(f"Epoch {i} train loss {train_loss[-1]} acc {train_acc[-1]} val loss {val_loss[-1]} acc {val_acc[-1]}")
    
    scheduler.step()

# Plotting the training/validation logs
fig, axes = plt.subplots(ncols=2, figsize=(15, 6))

axes[0].plot(train_loss, label="Training")
axes[0].plot(val_loss, label="Validation")
axes[0].legend()
axes[0].set_title("Loss log")

axes[1].plot(train_acc, label="Training")
axes[1].plot(val_acc, label="Validation")
axes[1].legend()
axes[1].set_title("Accuracy log")

plt.tight_layout()
plt.show()

# Predict function
def predict(img_path):
    img = transform(np.array(Image.open(img_path).convert("RGB"))).unsqueeze(0)
    best_model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            img = img.cuda()
        out = best_model(img)
        index = out.argmax(1).item()
        
    return index, out[0][index].item()

# Evaluate on test set (optional)
test = create_df(covid_path, normal_path)  # Assuming using the same directories for test
test["labels"] = le.transform(test["labels"].values)
test = test.sample(frac=1).reset_index(drop=True)

truth = []
probas = []
preds = []

for i in range(test.shape[0]):
    pred, proba = predict(test.iloc[i, 0])
    truth.append(test.iloc[i, 1])
    preds.append(pred)
    probas.append(proba * 100)

# Classification report and confusion matrix
print(classification_report(preds, truth))
sns.heatmap(confusion_matrix(preds, truth), annot=True, fmt='d')
plt.title(f"Accuracy: {round(accuracy_score(preds, truth) * 100, 2)}%")
plt.show()
