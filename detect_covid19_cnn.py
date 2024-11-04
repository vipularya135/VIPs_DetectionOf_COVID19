import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths for COVID and Normal images
path_covid = r'C:\Users\DELL\Desktop\covid19\Covid'
path_normal = r'C:\Users\DELL\Desktop\covid19\Normal'

# Function to create dataframe with images and labels
def create_df(covid_path, normal_path):
    dd = {"images": [], "labels": []}
    
    # Add COVID images
    for img in os.listdir(covid_path):
        dd["images"].append(os.path.join(covid_path, img))
        dd["labels"].append("covid")
    
    # Add Normal images
    for img in os.listdir(normal_path):
        dd["images"].append(os.path.join(normal_path, img))
        dd["labels"].append("normal")
        
    return pd.DataFrame(dd)

# Creating dataframe for images
df = create_df(path_covid, path_normal)

# Encoding labels
le = LabelEncoder()
df["labels"] = le.fit_transform(df["labels"].values)

# Splitting the data into train and validation sets
train, val = train_test_split(df.values, random_state=42, test_size=0.2)

# Data transformation pipeline
IMG_SIZE = 64
transform = transforms.Compose([transforms.ToPILImage(),
                               transforms.ToTensor(),
                               transforms.Resize((IMG_SIZE, IMG_SIZE)),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Custom Dataset class
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

# DataLoaders
BATCH = 32
train_ds = Pipeline(train, transform)
val_ds = Pipeline(val, transform)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)  # No activation here
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)  # No activation here
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)  # Adjusted size based on 64x64 input
        self.fc2 = nn.Linear(128, 2)  # 2 for binary classification (covid/normal)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply ReLU activation
        x = self.pool(F.relu(self.conv2(x)))  # Apply ReLU activation
        x = x.view(-1, 32 * 14 * 14)  # Flatten the tensor for fully connected layers
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = self.dropout(x)
        x = self.fc2(x)
        return nn.functional.softmax(x, dim=1)


# Training settings
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 20
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
    
    print("Epoch {} train loss {} acc {} val loss {} acc {}".format(i, train_loss[-1], train_acc[-1],
                                                                   val_loss[-1], val_acc[-1]))

# Plot training and validation results
fig, axes = plt.subplots(ncols=2, figsize=(15, 6))

index = 0

axes[index].plot(train_loss, label="Training")
axes[index].plot(val_loss, label="Validating")
axes[index].legend()
axes[index].set_title("Loss log")

index += 1

axes[index].plot(train_acc, label="Training")
axes[index].plot(val_acc, label="Validating")
axes[index].legend()
axes[index].set_title("Accuracy log")

plt.tight_layout()
plt.show()

# Prediction function for test data
def predict(img):
    img = transform(np.array(Image.open(img).convert("RGB"))).view([1, 3, 64, 64])
    best_model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            img = img.cuda()
        out = best_model(img)
        index = out.argmax(1).item()
        
    return index, out[0][index].item()

# Predicting test images and evaluation
truth = []
probas = []
preds = []

for i in range(val.shape[0]):
    pred, proba = predict(val[i, 0])
    truth.append(val[i, 1])
    preds.append(pred)
    probas.append(proba*100)

# Evaluating model performance
print(classification_report(preds, truth))
sns.heatmap(confusion_matrix(preds, truth), annot=True, fmt='d')
plt.title("Score: {}%".format(round(accuracy_score(preds, truth)*100, 2)))
plt.show()

# Visual inspection of predictions: Image, True Label, and Predicted Label
index = 0
truth_labels = le.inverse_transform(np.array(truth))
pred_labels = le.inverse_transform(np.array(preds))

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 6))

for i in range(3):
    for j in range(3):
        # Load the image to display
        img = Image.open(val[index, 0])
        
        # Display the image
        axes[i][j].imshow(np.array(img))
        
        # Set title with true and predicted labels
        axes[i][j].set_title("Predicted: {}\nTrue: {}".format(pred_labels[index], truth_labels[index]))
        
        index += 1

plt.tight_layout()
plt.show()
