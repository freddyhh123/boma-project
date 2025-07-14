import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import time

DATA_DIR = "dataset"
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 256
NUM_CLASSES = 3
RUNS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using: {DEVICE}")

model_list = [
    "convnextv2_tiny",               # Pure CNN
    #"swin_large_patch4_window12_384",# Transformer
    "efficientnetv2_rw_m",           # Pure CNN
    "swinv2_base_window8_256",       # Transformer - Bigger
    "edgenext_base",                 # Efficient hybrid + attention
    "swin_base_patch4_window7_224",   # Transformer
    "vit_base_patch16_224.orig_in21k_ft_in1k",  # Baseline ViT
    "beit_base_patch16_224",         # SSL transformer
    "mobilevitv2_100.cvnets_in1k",   # Lightweight transformer
]

# This specifies what size inputs need to be for each model
# The script can rescale to any size.
model_input_sizes = {
    "convnextv2_tiny": 224,
    "efficientnetv2_rw_m": 224,
    "swinv2_base_window8_256": 256,
    "edgenext_base": 224,
    "swin_base_patch4_window7_224": 224,
    "swin_large_patch4_window12_384": 384,
    "vit_base_patch16_224.orig_in21k_ft_in1k": 224,
    "beit_base_patch16_224": 224,
    "mobilevitv2_100.cvnets_in1k": 256
}


def train_loop(model_name, balanced_weight=False):
    IMG_SIZE = model_input_sizes[model_name]

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    model_results = pd.DataFrame(columns=[
    "model", "run", "test_acc", "train_acc", "train_loss",
    "f1_macro", "f1_info", "train_time"
    ])

    train = ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    test = ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE)

    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    if balanced_weight:
        labels = [label for _, label in train.samples]
        class_counts = Counter(labels)
        num_classes = len(train.classes)
        total = sum(class_counts.values())
        weights = [total / class_counts[i] for i in range(num_classes)]
        weights = weights / weights.sum() * num_classes
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights.to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optomizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_acc_list = []
    train_loss = []
    start_time = time.time()
    for run in range(RUNS):
        for epoch in range(EPOCHS):
            model.train()
            total_loss, correct, total,running_loss = 0,0,0,0.0
            
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
            epoch_loss = total_loss / total
            train_loss.append(epoch_loss)
            train_acc_list.append(acc)
            
            print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f} | Accuracy: {acc:.2%}")
        end_time = time.time()

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
        cr = classification_report(all_labels, all_preds, target_names=target_names,output_dict=True)
        print(cr)

        cm = confusion_matrix(all_labels,all_preds)
        plt.figure(figsize=(5,4))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(ticks=[0, 1, 2], labels=target_names, rotation=45)
        plt.yticks(ticks=[0, 1, 2], labels=target_names)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"cm_{model_name}.png")

        model_run =  {
            "model" : model_name,
            "run" : run,
            "test_acc" : test_acc,
            "train_acc" : sum(train_acc_list) / EPOCHS,
            "train_loss" : train_loss,
            "f1_macro" : cr['macro avg']['f1-score'],
            "f1_info" : cr,
            "train_time" : end_time - start_time
        }
        
        pd.concat([model_results, pd.DataFrame([model_run])], ignore_index=True)
    
    return model_results

results_df = pd.DataFrame()

for model in model_list:
    print(f"Loading {model}")
    model_results = train_loop(model)
    results_df = pd.concat([results_df, model_results], ignore_index=True)
    results_df.to_csv("model_results.csv")



