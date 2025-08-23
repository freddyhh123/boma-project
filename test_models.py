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
import torch.nn.functional as F
from timm.data import resolve_model_data_config, create_transform
from torch.utils.data import WeightedRandomSampler

from losses import FocalLoss
from model_freeze_utils import (
    freeze_all, set_norm_trainable, unfreeze_stage,
    llrd_groups, WarmupScheduler
)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"Using: {DEVICE}")

# This is for the discord webhook implementation
WEBHOOK = ""

DATA_DIR   = "dataset"
BATCH_SIZE = 8
MAX_EPOCHS = 100
PATIENCE   = 8
NUM_CLASSES = 3
RUNS = 5
# Weights for 2021 training
MODEL_WEIGHTS_PATH = "maxvit_large_new_focal.pth"

BASE_LR = 3e-5         
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 400      
GRAD_CLIP_NORM = 1.0   

os.makedirs("saved_models", exist_ok=True)

model_list = [
    #"convnextv2_tiny",               # Pure CNN
    #"efficientnetv2_rw_m",           # Pure CNN
    #"swinv2_base_window8_256",       # Transformer - Bigger
    #"edgenext_base",                 # Efficient hybrid + attention
    "maxvit_large_tf_512.in1k",
    #"swin_base_patch4_window7_224",   # Transformer
    #"vit_base_patch16_224.orig_in21k_ft_in1k",  # Baseline ViT
    #"beit_base_patch16_224",         # SSL transformer
    #"mobilevitv2_100.cvnets_in1k",   # Lightweight transformer
    #"swin_large_patch4_window12_384",# Transformer - Biggest,
    #"swinv2_large_window12to16_192to256.ms_in22k_ft_in1k"
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
    "mobilevitv2_100.cvnets_in1k": 256,
    "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k" : 256
}

def notify_discord(message):
    payload = {"content": message}
    try:
        requests.post(WEBHOOK, json=payload, timeout=5)
        print("Discord notification sent.")
    except Exception as e:
        print(f"Failed: {e}")

# Instantiates the custom sampler
# This aims to increase how often the model sees the abandoned class
def make_sampler(train_subset, num_classes=3):
    indices = train_subset.indices
    labels = [train_subset.dataset.samples[i][1] for i in indices]
    counts = Counter(labels)
    cls_w = {c: 1.0 / max(1, counts[c]) for c in range(num_classes)}
    weights = [cls_w[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# This is part of the custom weight initialization
# It takes the pretrained weights from timm and adds the custom transforms
# https://huggingface.co/docs/timm/en/quickstart?utm_source=chatgpt.com
def build_transforms(model_name: str, num_classes: int = 3):
    target = model_input_sizes[model_name]
    tmp = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    cfg = resolve_model_data_config(tmp)
    cfg['input_size'] = (3, target, target)
    cfg['crop_pct']   = 1.0

    timm_train = create_transform(**cfg, is_training=True, scale=(1.0, 1.0), ratio=(1.0, 1.0))
    timm_eval  = create_transform(**cfg, is_training=False)

    domain_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.30, hue=0.05),
        transforms.RandomAutocontrast(),
        transforms.RandomAdjustSharpness(1.0, p=0.3),
    ])

    train_tf = transforms.Compose([domain_aug, timm_train])
    test_tf  = timm_eval
    return train_tf, test_tf

def train_loop(model_name, balanced_weight=False, manual_weight=False, tuning_freeze=True, focal_loss=False):
    transform_train, transform_test = build_transforms(model_name, NUM_CLASSES)

    cols = ["model", "run", "test_acc", "train_acc", "train_loss",
            "f1_macro", "f1_info", "train_time", "epochs_ran"]
    model_results = pd.DataFrame(columns=cols)

    train = ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform_train)
    test  = ImageFolder(os.path.join(DATA_DIR, "test"),  transform=transform_test)

    # Split train -> train/val (80/20)
    val_size = int(0.2 * len(train))
    train_size = len(train) - val_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(train, [train_size, val_size], generator=generator)

    # Implement the custom sampler
    sampler = make_sampler(train_subset, NUM_CLASSES)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test,         batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    steps_per_epoch = max(1, len(train_loader))

    for run in range(RUNS):
        if tuning_freeze:
            model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
        else:
            model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

        # Various implementations of weight initalization (details in report)
        # Balanced weight automatically calculates the weighting based on the number of classes
        if balanced_weight:
            subset_indices = train_subset.indices
            labels = [train_subset.dataset.samples[i][1] for i in subset_indices]
            class_counts = Counter(labels)
            total = sum(class_counts.values())
            weights = [total / class_counts.get(i, 1) for i in range(NUM_CLASSES)]
            weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
        # Manual weight assigns class weighting manually
        elif manual_weight:
            manual_weights = torch.tensor([0.5, 1.5, 1.25], dtype=torch.float32, device=DEVICE)
            criterion = nn.CrossEntropyLoss(weight=manual_weights, label_smoothing=0.05)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        # Implement the custom focal loss
        if focal_loss:
            alpha_vec = torch.tensor([0.5, 2.2, 1.0], device=DEVICE)
            criterion = FocalLoss(alpha=alpha_vec, gamma=2.0, reduction='mean')

        scaler = GradScaler()

        # If training on 2021, load pre-trained weights and start unfreezing schedule
        if tuning_freeze:
            state = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
            model.load_state_dict(state, strict=True)
            freeze_all(model)
            set_norm_trainable(model, False)
            for p in model.head.parameters():
                p.requires_grad = True

            optimizer = torch.optim.AdamW(llrd_groups(model, base_lr=BASE_LR, wd=WEIGHT_DECAY), betas=(0.9, 0.999))
            tmax_steps = MAX_EPOCHS * steps_per_epoch
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax_steps)
            scheduler = WarmupScheduler(optimizer, warmup_steps=WARMUP_STEPS, after_scheduler=cosine)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
            tmax_steps = MAX_EPOCHS * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax_steps)

        best_f1 = 0.0
        patience_epochs = 0
        best_model_state = None
        train_acc_list, train_loss_list, val_loss_list = [], [], []

        start_time = time.time()
        for epoch in range(MAX_EPOCHS):
            model.train()
            total_loss, correct, total = 0.0, 0, 0

            if tuning_freeze and epoch in [2, 4, 6, 8]:
                stage_to_unfreeze = {2: 0, 4: 1, 6: 2, 8: 3}[epoch]
                unfreeze_stage(model, stage_to_unfreeze)
                if epoch >= 4:
                    set_norm_trainable(model, True)

                optimizer = torch.optim.AdamW(llrd_groups(model, base_lr=BASE_LR, wd=WEIGHT_DECAY), betas=(0.9, 0.999))
                epochs_left = MAX_EPOCHS - epoch
                tmax_steps = max(1, epochs_left * steps_per_epoch)
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax_steps)
                scheduler = WarmupScheduler(optimizer, warmup_steps=WARMUP_STEPS, after_scheduler=cosine)

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()

                if GRAD_CLIP_NORM is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                scaler.step(optimizer)
                scaler.update()

                scheduler.step()

                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / max(1, total)
            epoch_loss = total_loss / max(1, total)
            train_loss_list.append(epoch_loss)
            train_acc_list.append(train_acc)

            # Predict validation set for patience system
            model.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)

                    val_loss_sum += loss.item() * inputs.size(0)
                    val_correct  += (preds == labels).sum().item()
                    val_total    += labels.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_acc = val_correct / max(1, val_total)
            val_loss_epoch = val_loss_sum / max(1, val_total)
            val_loss_list.append(val_loss_epoch)
            cr = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
            f1_macro = cr['macro avg']['f1-score']

            print(f"[{model_name} | Run {run} | Epoch {epoch+1}] "
                  f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | "
                  f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss_epoch:.4f} | "
                  f"F1 Macro: {f1_macro:.4f}")

            if f1_macro > best_f1:
                best_f1 = f1_macro
                patience_epochs = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_epochs += 1

            if patience_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} for {model_name}")
                break

        end_time = time.time()

        # Save best state for every run
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            torch.save(model.state_dict(), f"saved_models/{model_name}_run{run}_best.pth")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = sum(int(p == l) for p, l in zip(all_preds, all_labels)) / max(1, len(all_preds))
        print(f"Test Accuracy: {test_acc:.2%}")

        target_names = ["non-settlement", "abandoned", "active"]
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

        row = {
            "model": model_name,
            "run": run,
            "test_acc": test_acc,
            "train_acc": sum(train_acc_list) / max(1, len(train_acc_list)),
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "f1_macro": f1_macro,
            "f1_info": cr,
            "train_time": end_time - start_time,
            "epochs_ran": epoch + 1
        }
        model_results = pd.concat([model_results, pd.DataFrame([row])], ignore_index=True)

    return model_results

notify_discord("Training Started")

results_df = pd.DataFrame()
for model in model_list:
    print(f"\n Loading {model} ")
    model_results = train_loop(model, focal_loss=True, tuning_freeze=True)
    results_df = pd.concat([results_df, model_results], ignore_index=True)
    results_df.to_csv("model_results.csv", index=False)

notify_discord("Training Finished")
