import os
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
from timm import create_model
from tqdm import tqdm


class BomaDataset(Dataset):
    def __init__(self, patch_dir, csv_file, transform=None):
        self.patch_dir = patch_dir
        self.patches_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return(len(self.patches_df))
    
    def __getitem__(self, idx):
        img_name = self.patches_df.iloc[idx]["filename"]
        img_path = os.path.join(self.patch_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name
    

model = create_model("maxvit_large_tf_512.in1k", pretrained=False, num_classes=3)
model.load_state_dict(torch.load(r"maxvit_large_tf_512.in1k_run4_best.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])


patch_dir = "2013_patches"
csv_file = os.path.join("patch_metadata_new.csv")

patch_data = BomaDataset(patch_dir, csv_file, transform=transform)
dataloader = DataLoader(patch_data, batch_size=32, shuffle=False)

results = []

label_mapping = {
    0: "non-settlement",
    1: "active",
    2: "abandoned"
}

with torch.no_grad():
    for images, filenames in tqdm(dataloader,desc="Classifying Patches"):
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs,dim=1).cpu().numpy()
        for fname, pred in zip(filenames,preds):
            results.append({"filename": fname, "label": label_mapping[pred]})


patch_df = pd.read_csv(csv_file)
preds_df = pd.DataFrame(results)

df = patch_df.merge(preds_df, on="filename")

geometry = [box(xmin,ymin,xmax,ymax) for xmin,ymin,xmax,ymax in zip(
    df['xmin'], df['ymin'], df['xmax'], df['ymax']
)]

gdf = gpd.GeoDataFrame(df, geometry=geometry)
gdf.set_crs(epsg=32637, inplace=True)


gdf.to_file("patch_predictions.gpkg",layer="settlement_patches",driver="GPKG")

gdf.plot(column='label', legend=True, figsize=(10, 10))
plt.title('Settlement Type by Patch')
plt.savefig("graph.png")