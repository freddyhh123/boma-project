import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import ast
import json
import os

OUTPUT_DIR = "graphs"
MODEL_FOLDER = None
CLASSES = ['non-settlement', 'abandoned', 'active']
GPKG_YEAR1 = "patch_predictions(2).gpkg"
GPKG_YEAR2 = "patch_predictions_maxvit_2021.gpkg"
YEAR1, YEAR2 = 2013, 2021
LABEL_FIELD = "label"  
ABANDONED_POLICY = "drop" 
SETTLEMENT_VALUE = "active" 
EPS_M_OVERRIDE = None               
MIN_SAMPLES = 3        
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_label(s: pd.Series, policy: str) -> pd.Series:
    s = s.astype("string").str.strip().str.lower()
    if policy == "merge_to_active":
        s = s.replace({"abandoned": "active"})
    elif policy == "drop":
        s = s.where(~s.eq("abandoned"), other=pd.NA)
    return s

def estimate_tile_width_m(gdf: gpd.GeoDataFrame) -> float:
    if gdf.empty:
        return 0.0
    b = gdf.geometry.bounds
    widths = (b["maxx"] - b["minx"]).to_numpy()
    return float(np.median(widths))

def to_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["geometry"] = out.geometry.centroid
    return out

def run_dbscan(points: gpd.GeoDataFrame, eps_m: float, min_samples: int) -> gpd.GeoDataFrame:
    if points.empty:
        pts = points.copy()
        pts["cluster"] = np.array([], dtype=int)
        return pts
    coords = np.vstack([points.geometry.x.values, points.geometry.y.values]).T
    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(coords)
    out = points.copy()
    out["cluster"] = labels
    return out

def summarize(pts: gpd.GeoDataFrame) -> dict:
    total = int(len(pts))
    clustered_tiles = int((pts["cluster"] >= 0).sum())
    n_clusters = int(pts["cluster"].max() + 1) if clustered_tiles > 0 else 0
    pct = (clustered_tiles / total * 100.0) if total else 0.0
    return {
        "total_tiles": total,
        "tiles_in_clusters": clustered_tiles,
        "pct_tiles_in_clusters": pct,
        "clusters": n_clusters
    }

def plot_clusters(poly1, pts1, poly2, pts2, year1, year2, eps_m):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    plt.subplots_adjust(wspace=0.2)

    ax1 = axes[0]
    if not poly1.empty:
        poly1.plot(ax=ax1, facecolor="none", edgecolor="lightgray", linewidth=0.3)
    if not pts1.empty:
        pts1[pts1["cluster"] == -1].plot(ax=ax1, markersize=6, color="lightgray")
        pts1[pts1["cluster"] >= 0].plot(ax=ax1, column="cluster", categorical=True,
                                        markersize=8, cmap="tab20")
    ax1.set_aspect("equal")
    ax1.set_title(f"{year1}")
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = axes[1]
    if not poly2.empty:
        poly2.plot(ax=ax2, facecolor="none", edgecolor="lightgray", linewidth=0.3)
    if not pts2.empty:
        pts2[pts2["cluster"] == -1].plot(ax=ax2, markersize=6, color="lightgray")
        pts2[pts2["cluster"] >= 0].plot(ax=ax2, column="cluster", categorical=True,
                                        markersize=8, cmap="tab20")
    ax2.set_aspect("equal")
    ax2.set_title(f"{year2}")
    ax2.set_yticks([])
    ax2.set_xlabel("Easting (m)")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=45)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())

    fig.suptitle("Active Settlement Clusters", y=0.95)

    plt.show()

def compute_clusters():
    g1 = gpd.read_file(GPKG_YEAR1)
    g2 = gpd.read_file(GPKG_YEAR2)

    g1["_lab"] = clean_label(g1[LABEL_FIELD], ABANDONED_POLICY)
    g2["_lab"] = clean_label(g2[LABEL_FIELD], ABANDONED_POLICY)
    if ABANDONED_POLICY == "drop":
        g1 = g1.dropna(subset=["_lab"])
        g2 = g2.dropna(subset=["_lab"])

    sett1 = g1[g1["_lab"] == SETTLEMENT_VALUE].copy()
    sett2 = g2[g2["_lab"] == SETTLEMENT_VALUE].copy()

    tile_w1 = estimate_tile_width_m(sett1)
    tile_w2 = estimate_tile_width_m(sett2)
    tile_w = np.nanmedian([v for v in [tile_w1, tile_w2] if v and not np.isnan(v)]) if (tile_w1 or tile_w2) else 0.0
    eps_m = float(EPS_M_OVERRIDE) if EPS_M_OVERRIDE else (1.5 * tile_w if tile_w > 0 else 300.0)

    p1 = to_centroids(sett1)
    p2 = to_centroids(sett2)
    p1c = run_dbscan(p1, eps_m, MIN_SAMPLES)
    p2c = run_dbscan(p2, eps_m, MIN_SAMPLES)
    s1 = summarize(p1c)
    s2 = summarize(p2c)

    print(f"DBSCAN eps = {eps_m:.1f} m, min_samples = {MIN_SAMPLES}, estimated tile width ≈ {tile_w:.1f} m\n")
    print(f"{YEAR1}: {s1}")
    print(f"{YEAR2}: {s2}")

    plot_clusters(sett1, p1c, sett2, p2c, YEAR1, YEAR2, eps_m)

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

# plot_data("model_results-normal.csv")
compute_clusters()
    


