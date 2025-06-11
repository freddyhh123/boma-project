import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
import numpy as np
import os
from random import uniform
from PIL import Image
import json

SHAPEFILE = "Data/Boma Project/HF12032019.shp"
RASTER = "Data/Boma Project/Mosaic201311.tif"
PATCH_SIZE = 256
LABEL_COLUMN = "Status"
OUTPUT_DIR = "patches"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def random_point_within(bounds):
    minx, miny, maxx, maxy = bounds.left, bounds.bottom, bounds.right, bounds.top
    x = uniform(minx, maxx)
    y = uniform(miny, maxy)
    return Point(x, y)

def save_patch(src,x,y,label,idx,out_dir):
    try:
        row_pix, col_pix = rowcol(src.transform, x, y)
    except Exception as e:
        print(e)
        return None
    
    half = PATCH_SIZE // 2
    if col_pix - half < 0 or row_pix - half < 0 or col_pix + half >=src.width or row_pix + half >= src.height:
        return None
    
    window = Window(col_pix - half, row_pix - half, PATCH_SIZE, PATCH_SIZE)
    try:
        patch = src.read(window=window)
        if patch.shape[1] != PATCH_SIZE or patch.shape[2] != PATCH_SIZE:
            return None
    except Exception:
        print("Error in image generation")
        return None
    
    patch = np.transpose(patch, (1,2,0))
    patch = np.clip(patch,0,255).astype(np.uint8)
    img = Image.fromarray(patch)
    filename = f"{idx}_{label}.png"
    img.save(os.path.join(out_dir,filename))
    
    return {
        "image" : os.path.join(out_dir,filename),
        "label" : label,
        "x":x,
        "y":y,
        "row_pix": row_pix,
        "col_pix": col_pix
    }
    
shapefile_gdf = gpd.read_file(SHAPEFILE)
images = []
missing_labels = []
label_counts = {0: 0, 1: 0, 2: 0}

with rasterio.open(RASTER) as src:
    raster_crs = src.crs
    shapefile_gdf = shapefile_gdf.to_crs(raster_crs)
    shapefile_gdf["centroid"] = shapefile_gdf.geometry.centroid
    shapefile_gdf = shapefile_gdf.dropna(subset=["centroid"])
    
    for idx, row in shapefile_gdf.iterrows():
        label_raw = row[LABEL_COLUMN]
        if pd.isna(label_raw):
            print(f"No label at idx: {idx}")
            missing_labels.append(idx)
            continue
        
        label = str(label_raw).strip().lower()

        if label in ["occupied", "permanent", "seasonal", "temporally", "campsite", "lodge"]:
            label = 2 
        elif label == "abandoned":
            label = 1
        elif label == "agriculture":
            label = 0
        else:
            print(f"Unknown label: {label}")
            continue
            
        x,y = row["centroid"].x, row["centroid"].y
        out = save_patch(src,x,y,label,idx,OUTPUT_DIR)
        if out:
            out["geometry_wkt"] = row.geometry.wkt
            images.append(out)
            label_counts[label] +=1
    
    print("Onto non-settlement images")
    
    buffered = shapefile_gdf.buffer(100)
    settlement_zone = unary_union(buffered)
    
    non_settlement_points = []
    attempts = 0
    
    print("Raster bounds:", src.bounds)
    print("Shapefile extent:", shapefile_gdf.total_bounds)
    
    while len(non_settlement_points) < 4700 and attempts < 20000:
        pt = random_point_within(src.bounds)
        if not settlement_zone.contains(pt):
            non_settlement_points.append(pt)
        attempts += 1
    
    for i,pt in enumerate(non_settlement_points):
        out = save_patch(src,pt.x,pt.y,label=0,idx=f"{i}",out_dir=OUTPUT_DIR)
        if out:
            out["geometry_wkt"] = pt.wkt
            out["x"] = x
            out["y"] = y
            images.append(out)
            label_counts[0] += 1
    
    df = pd.DataFrame(images)
    df.to_csv(os.path.join(OUTPUT_DIR, "patches_metadata.csv"), index=False)
    
    print(f"Final patch counts:")
    print(f" - Non-settlement: {label_counts[0]}")
    print(f" - Abandoned: {label_counts[1]}")
    print(f" - Active: {label_counts[2]}")
    print(f" - Missing labels in shapefile: {len(missing_labels)}")
            
