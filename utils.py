import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

DATASET_DIR = "dataset/train"
BASE_FOLDER = "dataset"
LABELS = ["0","1","2"]

def split_patches(patch_folder):
    df = pd.read_csv(f"{patch_folder}/patches_metadata.csv")
    train_df,test_df = train_test_split(df, test_size=0.2,stratify=df["label"],random_state=420)
    
    for split, subset in [("train", train_df), ("test",test_df)]:
        for _, row in subset.iterrows():
            label = str(row["label"])
            src = row["image"]
            dst_dir = os.path.join(BASE_FOLDER, split, label)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(src))
            shutil.copy(src,dst)

def get_random_images(label_dir):
    files = [f for f in os.listdir(label_dir) if f.endswith(".png")]
    if not files:
        return None
    file = random.choice(files)
    path = os.path.join(label_dir, file)
    return Image.open(path), os.path.basename(path)

def show_random_set():
    fig, axs = plt.subplots(1, len(LABELS),figsize=(5 * len(LABELS),5))
    fig.suptitle("Press and key to show a new random set", fontsize=14)
    
    for ax, label in zip(axs, LABELS):
        label_dir = os.path.join(DATASET_DIR,label)
        img, fname = get_random_images(label_dir)
        if img:
            ax.imshow(img)
            ax.set_title(f"Label: {label}\n{fname}", fontsize=12)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, f"No images for label {label}", ha='center', va='center')
            ax.axis("off")
    plt.tight_layout()
    return fig

def viewer_loop():
    while True:
        fig = show_random_set()
        pressed = plt.waitforbuttonpress()
        plt.close(fig)
        if pressed is None:
            break


#split_patches("patches")
#viewer_loop()