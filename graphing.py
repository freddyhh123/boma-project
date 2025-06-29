import pandas as pd
import matplotlib.pyplot as plt
import ast
import json
import os

OUTPUT_DIR = "graphs"
MODEL_FOLDER = None
CLASSES = ['non-settlement', 'abandoned', 'active']
os.makedirs(OUTPUT_DIR, exist_ok=True)

def count_examples():
    df = pd.read_csv("patches/patches_metadata.csv")
    df["label"].value_counts().sort_index().plot(kind='bar', color='lightcoral',zorder=3)
    plt.title("Label Distribution in Patch Dataset")
    plt.xlabel("Label (0 = No settlement, 1 = Abandoned, 2 = Active)")
    plt.ylabel("Number of Patches")
    plt.xticks(rotation=0)
    plt.grid(axis='y', zorder=0)
    plt.tight_layout()
    plt.show()

def accuracy_plot(data):
    plt.figure()
    x = range(len(data))
    plt.bar(x, data["train_acc"], width=0.4, label="Train Accuracy", align="center")
    plt.bar([i + 0.4 for i in x], data["test_acc"], width=0.4, label="Test Accuracy", align="center")
    plt.xticks([i + 0.2 for i in x], data["model_name_short"],rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Train v Test Accuracy")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, "accuracy.png"))
    plt.close()

def loss_curve(data):
    plt.figure()
    for i, row in data.iterrows():
        if row["train_loss"] == []:
            continue
        plt.plot(row["train_loss"],label=row["model_name_short"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, "loss.png"))
    plt.close()

def f1_comparison(data):
    plt.figure()
    plt.bar(data["model_name_short"], data["f1_macro"])
    plt.ylabel("F1 Macro")
    plt.title("F1 Macro")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, "f1.png"))
    plt.close()

def training_time(data):
    plt.figure()
    plt.bar(data["model_name_short"],data["train_time"])
    plt.ylabel("Training Time (s)")
    plt.title("Training time per model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, "training_time.png"))
    plt.close()

def per_class_f1(data):
    for cls in CLASSES:
        f1_scores = [row["f1_info"][cls]["f1-score"] for _, row in data.iterrows()]
        plt.plot(data["model_name_short"], f1_scores, label=cls)
    plt.title("Per-class F1")
    plt.ylabel("F1 Score")
    plt.xlabel("Model")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, "f1_class.png"))
    plt.close()

def per_class_recall(data):
    for cls in CLASSES:
        f1_scores = [row["f1_info"][cls]["recall"] for _, row in data.iterrows()]
        plt.plot(data["model_name_short"], f1_scores, label=cls)
    plt.title("Per-class Recall")
    plt.ylabel("Recall")
    plt.xlabel("Model")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, "recall.png"))
    plt.close()

def plot_data(filename):
    global MODEL_FOLDER
    file_basename = os.path.splitext(os.path.basename(filename))[0]
    MODEL_FOLDER = os.path.join(OUTPUT_DIR, file_basename)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    df = pd.read_csv(filename)
    df["train_loss"] = df["train_loss"].apply(ast.literal_eval)
    df['f1_info'] = df['f1_info'].apply(ast.literal_eval)
    df['model_name_short'] = df['model'].apply(lambda x: x.split('_')[0])
    df = df.sort_values("model_name_short").reset_index(drop=True)
    accuracy_plot(df)
    loss_curve(df)
    f1_comparison(df)
    training_time(df)
    per_class_f1(df)
    per_class_recall(df)

plot_data("model_results_10_epoch2.csv")

    


