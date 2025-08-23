import pandas as pd
import numpy as np
import rasterio.windows
from sklearn.model_selection import train_test_split
import shutil
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATASET_DIR = "dataset/train"
BASE_FOLDER = "dataset"
LABELS = ["0","1","2"]

# Function that splits patches into test/train
def split_patches(patch_folder):
    df = pd.read_csv(f"{patch_folder}/patches_metadata.csv")
    if "include" in df:
        df = df[df["include"] != False]
    train_df,test_df = train_test_split(df, test_size=0.2,stratify=df["label"],random_state=420)
    
    for split, subset in [("train", train_df), ("test",test_df)]:
        for _, row in subset.iterrows():
            label = str(row["label"])
            src = row["image"]
            dst_dir = os.path.join(BASE_FOLDER, split, label)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(src))
            shutil.copy(src,dst)

# Helper function to check for patches marked as bad and drop them from metadata file
def clean_patch_metadata(patch_folder, bad_patch_folder):
    file = os.path.join(patch_folder, "patches_metadata.csv")
    df = pd.read_csv(file)
    df = df.drop("is_good_patch",axis=1)
    bad_patches = set(os.listdir(bad_patch_folder))
    df["include"] = df["image"].apply(
        lambda x: os.path.basename(x).replace("patches/", "") not in bad_patches
    )
    df.to_csv(file,index=False)
  
# Functions to show random images for dataset reviewing  
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

# Split entire mosaic into patches for analysis, and write metadata file
def process_patches(path_2013,path_2021,output_2021_dir,output_2013_dir, patch_size=256, overlap=0):
    os.makedirs(output_2021_dir, exist_ok=True)
    os.makedirs(output_2013_dir, exist_ok=True)

    metadata = []
    
    with rasterio.open(path_2021) as src_2021, rasterio.open(path_2013) as ref_2013:
        assert src_2021.crs == ref_2013.crs, "CRS mismatch"
        assert src_2021.transform == ref_2013.transform, "Transform mismatch"
        assert src_2021.width == ref_2013.width and src_2021.height == ref_2013.height, "Dimension mismatch"
        
        width = src_2021.width
        height = src_2021.height
        bands = src_2021.count
        transform = src_2021.transform
        
        for row in range(0,height,patch_size-overlap):
            for col in range(0,width,patch_size - overlap):
                window = Window(col_off=col, row_off=row,width=patch_size,height=patch_size).round_offsets()
                
                if window.col_off + window.width > width:
                    window = Window(window.col_off, window.row_off, width - window.col_off,window.height)
                if window.row_off + window.height > height:
                    window = Window(window.col_off,window.row_off,window.width,height - window.row_off)
        
                patch_2021 = src_2021.read(window=window)
                patch_2013 = ref_2013.read(window=window)
                                
                patch_name = f"patch_r{int(window.row_off)}_c{int(window.col_off)}.tif"
                
                patch_transform = rasterio.windows.transform(window,transform)
                profile = src_2021.profile.copy()
                profile.update({
                    "height": int(window.height),
                    "width": int(window.width),
                    "transform": patch_transform,
                    "dtype": "uint8"
                })

                with rasterio.open(os.path.join(output_2021_dir, patch_name), 'w', **profile) as dst:
                    dst.write(patch_2021.astype(np.float32))
                with rasterio.open(os.path.join(output_2013_dir, patch_name), 'w', **profile) as dst:
                    dst.write(patch_2013.astype(np.float32))
                    
                xmin,ymax = patch_transform * (0,0)
                xmax, ymin = patch_transform * (window.width, window.height)
                
                metadata.append({
                    "filename": patch_name,
                    "row_off": int(window.row_off),
                    "col_off": int(window.col_off),
                    "width": int(window.width),
                    "height": int(window.height),
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                })
                print(f"Saved: {patch_name} to both folders")
    
    df = pd.DataFrame(metadata)
    df.to_csv("patch_metadata.csv",index=False)
    
# Match mosaics to be same size 
def resize_mosaic():
    with rasterio.open("Data/Boma Project/Mosaic201311.tif") as ref:
        dst_height = ref.height
        dst_width = ref.width
        dst_transform = ref.transform
        dst_crs = ref.crs

    with rasterio.open("Data/2021/2021_unprocessed.tif") as src:
        profile = src.profile.copy()
        data = np.empty((src.count, dst_height, dst_width), dtype=np.uint8)

        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=data[i-1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )

        profile.update({
            'height': dst_height,
            'width': dst_width,
            'transform': dst_transform,
            'crs': dst_crs
        })

    with rasterio.open("2021_reformatted.tif", "w", **profile) as dst:
        dst.write(data)

# Helper function to rename and add new patches
def create_updated_metadata(original_path,new_path,output_path):
    df = pd.read_csv(original_path)
    new_filenames = set(os.listdir(new_path))
    df["filename_only"] = df["image"].apply(lambda x: os.path.basename(x))
    filtered_df = df[df["filename_only"].isin(new_filenames)].copy()
    filtered_df["image"] = filtered_df["filename_only"].apply(lambda x: os.path.join(new_path,x))
    filtered_df.drop(columns=["filename_only"],inplace=True)
    filtered_df.to_csv(output_path,index=False)

# Mistake in dataset reviewer required this function to flip labels on filenames
def flip_filenames():
    folder_path = "2021_training_patches"

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            parts = filename.rsplit('_', 1)  # split into id and label
            if len(parts) == 2:
                id_part, label_ext = parts
                label = label_ext.replace('.png', '')

                # Check if label is 1 or 2
                if label == '1':
                    new_label = '2'
                elif label == '2':
                    new_label = '1'
                else:
                    continue  # Skip if label is not 1 or 2

                new_filename = f"{id_part}_{new_label}.png"

                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)

                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    #compute_mean_std("dataset")
    #flip_filenames()
    #clean_patch_metadata("patches","bad_patches")
    split_patches("patches")
    #viewer_loop()

    #resize_mosaic()
    #process_patches(
    #"Clipped/2013_clipped.tif",
    # "Clipped/2021_clipped.tif",
    #  "output/2021_patches",
    #   "output/2013_patches",
    #   256
    #)

    #create_selected_metadata("2021_patches/patches_metadata.csv","2021_training_patches","2021_training_patches/patches_metadata.csv")