import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

MODEL_NAME = "efficientnet_b2"
DATA_DIR = "dataset"
BATCH_SIZE = 32
EPOCHS = 5
IMG_SIZE = 256
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using: {DEVICE}")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

train = ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
test = ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test, batch_size=BATCH_SIZE)

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optomizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_acc_list = []
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0,0,0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optomizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optomizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    acc = correct / total
    avg_loss = total_loss / total
    train_acc_list.append(acc)
    
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Accuracy: {acc:.2%}")
    
model.eval()
all_preds, all_labels = [],[]
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs,labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
test_acc = sum(int(p == l) for p, l in zip(all_preds, all_labels)) / len(all_preds)
print(f"Test Accuracy: {test_acc:.2%}") 

target_names = ["non-settlement", "abandoned", "active"]
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=target_names))

cm = confusion_matrix(all_labels,all_preds)
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix for {MODEL_NAME}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(ticks=[0, 1, 2], labels=target_names, rotation=45)
plt.yticks(ticks=[0, 1, 2], labels=target_names)
plt.colorbar()
plt.tight_layout()
plt.savefig(f"cm_{MODEL_NAME}.png")