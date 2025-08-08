import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from sklearn.metrics import classification_report
import pandas as pd
from collections import Counter
import time
import copy
import requests
from torch.cuda.amp import autocast, GradScaler

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

DATA_DIR = "dataset"
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
IMG_SIZE = 256
NUM_CLASSES = 3
RUNS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_WEIGHTS_PATH = "maxvit_large_tf_512.in1k_run4_best.pth"
print(f"Using: {DEVICE}")

os.makedirs("saved_models", exist_ok=True)

model_list = [
    #"convnextv2_tiny",               # Pure CNN
    #"efficientnetv2_rw_m",           # Pure CNN
    #"swinv2_base_window8_256",       # Transformer - Bigger
    #"edgenext_base",                 # Efficient hybrid + attention
    #"maxvit_large_tf_512.in1k",
    #"swin_base_patch4_window7_224",   # Transformer
    #"vit_base_patch16_224.orig_in21k_ft_in1k",  # Baseline ViT
    #"beit_base_patch16_224",         # SSL transformer
    #"mobilevitv2_100.cvnets_in1k",   # Lightweight transformer
    #"swin_large_patch4_window12_384",# Transformer - Biggest,
]

# This specifies what size inputs need to be for each model
# The script can rescale to any size.
model_input_sizes = {
    "maxvit_large_tf_512.in1k": 512,
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

def train_loop(model_name, balanced_weight=False, manual_weight=False, tuning_freeze=False):
    IMG_SIZE = model_input_sizes[model_name]

    # Reformat the images depending on the model
    transform_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    model_results = pd.DataFrame(columns=[
    "model", "run", "test_acc", "train_acc", "train_loss",
    "f1_macro", "f1_info", "train_time", "epochs_ran"
    ])

    train = ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform_train)
    test = ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform_test)

    # Make a validation set out of the train set (20%)
    val_size = int(0.2* len(train))
    train_size = len(train) - val_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_subset,batch_size=BATCH_SIZE)

    for run in range(RUNS):
        if tuning_freeze == False:
            model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
            model.to(DEVICE)
        else:
            model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
            model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
            model.to(DEVICE)
            for name, param in model.named_parameters():
                if "classifier" not in name and "fc" not in name and "head" not in name:
                    param.requires_grad = False
                    
        # Implemtation of balanced weighting to try and increase active/abandoned performance WIP
        if balanced_weight:
            subset_indices = train_subset.indices
            labels = [train.dataset.samples[i][1] for i in subset_indices]
            class_counts = Counter(labels)
            num_classes = NUM_CLASSES
            total = sum(class_counts.values())
            weights = [total / class_counts[i] for i in range(num_classes)]
            weights = weights / weights.sum() * num_classes
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights.to(DEVICE)
            criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
        if manual_weight:
            manual_weights = torch.tensor([0.5, 1.5, 1.5], dtype=torch.float32).to(DEVICE)
            criterion = torch.nn.CrossEntropyLoss(weight=manual_weights, label_smoothing=0.05)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        scaler = GradScaler()
        optomizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optomizer, T_max=MAX_EPOCHS)

        best_f1 = 0.0
        patience_epochs = 0
        best_model_state = None
        train_acc_list = []
        train_loss_list = []
        val_loss_list = []

        start_time = time.time()
        for epoch in range(MAX_EPOCHS):
            model.train()
            total_loss, correct, total = 0,0,0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                optomizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optomizer)
                scaler.update()
                scheduler.step()
                
                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            train_acc = correct / total
            epoch_loss = total_loss / total
            train_loss_list.append(epoch_loss)
            train_acc_list.append(train_acc)
            
            # The validation set is used for early stopping
            model.eval()
            val_loss, val_correct, val_total = 0,0,0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs,labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)

                    val_loss += loss.item() * inputs.size(0)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_acc = val_correct / val_total
            val_loss_epoch = val_loss / val_total
            val_loss_list.append(val_loss_epoch)
            cr = classification_report(all_labels, all_preds, output_dict=True)
            f1_macro = cr['macro avg']['f1-score']

            print(f"[Run {run} | Epoch {epoch+1}] "
            f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | "
            f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss_epoch:.4f}| F1 Macro: {f1_macro:.4f}")
        
            # Check if this epoch is the best if so then replace
            if f1_macro > best_f1:
                best_f1 = f1_macro
                patience_epochs = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_epochs += 1

            if patience_epochs >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1} for {model_name}")
                break

        end_time = time.time()

        # Save the best one
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            torch.save(model.state_dict(), f"saved_models/{model_name}_run{run}_best.pth")

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
        print(classification_report(all_labels, all_preds, target_names=target_names))

        model_run =  {
            "model" : model_name,
            "run" : run,
            "test_acc" : test_acc,
            "train_acc" : sum(train_acc_list) / len(train_acc_list),
            "train_loss" : train_loss_list,
            "val_loss": val_loss_list,
            "f1_macro" : cr['macro avg']['f1-score'],
            "f1_info" : cr,
            "train_time" : end_time - start_time,
            "epochs_ran" : epoch + 1
        }
        
        model_results = pd.concat([model_results, pd.DataFrame([model_run])], ignore_index=True)
    
    return model_results

results_df = pd.DataFrame()

for model in model_list:
    print(f"Loading {model}")
    model_results = train_loop(model,manual_weight=True)
    results_df = pd.concat([results_df, model_results], ignore_index=True)
    results_df.to_csv("model_results.csv", index=False)