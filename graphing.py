import pandas as pd
import matplotlib.pyplot as plt
import ast
import json
import os

OUTPUT_DIR = "graphs"
MODEL_FOLDER = None
CLASSES = ['non-settlement', 'abandoned', 'active']
os.makedirs(OUTPUT_DIR, exist_ok=True)


def prepare_data(filename):
    global MODEL_FOLDER
    file_basename = os.path.splitext(os.path.basename(filename))[0]
    MODEL_FOLDER = os.path.join(OUTPUT_DIR, file_basename)
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    df = pd.read_csv(filename)
    df["train_loss"] = df["train_loss"].apply(ast.literal_eval)
    df["f1_info"] = df["f1_info"].apply(ast.literal_eval)
    df["model_name_short"] = df["model"].apply(lambda x: x.split("_")[0])
    return df

def aggregate_data(df):
    agg = df.groupby("model_name_short").agg({
        "train_acc": ["mean","std"],
        "test_acc": ["mean","std"],
        "f1_macro": ['mean', 'std'],
        "train_time": ["mean"],
        "epochs_ran": ["mean"]
    })
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    return agg.reset_index()

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
    plt.bar(x, data["train_acc_mean"], yerr=data["train_acc_std"], width=0.4, label="Train Accuracy", align="center")
    plt.bar([i + 0.4 for i in x], data["test_acc_mean"], yerr=data["test_acc_std"], width=0.4, label="Test Accuracy", align="center")
    plt.xticks([i + 0.2 for i in x], data["model_name_short"], rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Train vs Test Accuracy (mean ± std across runs)")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, "accuracy.png"))
    plt.close()

def loss_curve(data):
    plt.figure()
    for model_name, group in data.groupby('model_name_short'):
        for _, row in group.iterrows():
            if row["train_loss"]:
                plt.plot(row["train_loss"], alpha=0.3)
        max_len = max(len(loss) for loss in group["train_loss"] if loss)
        avg_loss = [sum(loss[i] for loss in group["train_loss"] if len(loss) > i) / 
                    sum(1 for loss in group["train_loss"] if len(loss) > i)
                    for i in range(max_len)]
        plt.plot(avg_loss, label=f"{model_name} (avg)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves (faint = individual runs, bold = avg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, "loss.png"))
    plt.close()

def f1_comparison(data):
    plt.figure()
    plt.bar(data["model_name_short"], data["f1_macro_mean"], yerr=data["f1_macro_std"])
    plt.ylabel("F1 Macro")
    plt.title("F1 Macro (mean ± std across runs)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, "f1.png"))
    plt.close()
    
def training_time(data):
    plt.figure()
    plt.bar(data["model_name_short"], data["train_time_mean"])
    plt.ylabel("Training Time (s)")
    plt.title("Average Training Time per Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, "training_time.png"))
    plt.close()

def per_class_metrics(df, metric):
    df_class = {cls: [] for cls in CLASSES}
    grouped = df.groupby('model_name_short')
    for model, group in grouped:
        for cls in CLASSES:
            scores = [row["f1_info"][cls][metric] for _, row in group.iterrows()]
            df_class[cls].append((model, sum(scores) / len(scores)))
    for cls in CLASSES:
        models, scores = zip(*df_class[cls])
        plt.plot(models, scores, marker='o', label=cls)
    plt.title(f"Per-Class {metric.capitalize()} (mean across runs)")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Model")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, f"{metric}_class.png"))
    plt.close()

def plot_data(filename):
    df = prepare_data(filename)
    df_agg = aggregate_data(df)
    accuracy_plot(df_agg)
    loss_curve(df)
    f1_comparison(df_agg)
    training_time(df_agg)
    per_class_metrics(df, 'f1-score')
    per_class_metrics(df, 'recall')

plot_data("model_results-final.csv")

    


