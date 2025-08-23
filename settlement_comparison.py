import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np

gpkg_year1 = "patch_predictions(2).gpkg" 
gpkg_year2 = "patch_predictions_maxvit_2021.gpkg"
out_gpkg = "settlement_change_no_abandoned.gpkg"
year1, year2 = 2013, 2021

# How to handle abandoned boma (as they arent as reliable as the others) options: "merge_to_active" | "keep" | "drop"
abandoned_policy = "merge_to_active"

join_on = "filename"

def clean_label(s: pd.Series, policy: str) -> pd.Series:
    s = s.astype("string").str.strip().str.lower()
    if policy == "merge_to_active":
        s = s.replace({"abandoned": "active"})
    elif policy == "drop":
        s = s.where(~s.eq("abandoned"), other=pd.NA)
    return s

def round_df(df, ndigits=3, cols=None):
    out = df.copy()
    if cols is None:
        num_cols = out.select_dtypes(include="number").columns
    else:
        num_cols = cols
    out[num_cols] = out[num_cols].round(ndigits)
    return out

g1 = gpd.read_file(gpkg_year1)
g2 = gpd.read_file(gpkg_year2)

need = (["filename"] if (isinstance(join_on, str) and join_on=="filename") else ["row_off","col_off"]) + ["label","geometry"]
for df, nm in [(g1,"year1"), (g2,"year2")]:
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{nm} is missing columns: {missing}")

g1 = g1[need].copy()
g2 = g2[need].copy()

g1[f"label_{year1}"] = clean_label(g1["label"], abandoned_policy)
g2[f"label_{year2}"] = clean_label(g2["label"], abandoned_policy)
if abandoned_policy == "drop":
    g1 = g1.dropna(subset=[f"label_{year1}"])
    g2 = g2.dropna(subset=[f"label_{year2}"])

# Join the two years on the key selected earlier (filename in this case)
key_cols = [join_on] if isinstance(join_on, str) else join_on
left  = g1.drop(columns=["label"])
right = g2.drop(columns=["label","geometry"])
joined = left.merge(right, on=key_cols, how="inner", suffixes=(f"_{year1}", f"_{year2}"))
joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=g1.crs)

# Produce transition column
l1, l2 = f"label_{year1}", f"label_{year2}"
joined["transition"] = joined[l1].fillna("unknown") + "→" + joined[l2].fillna("unknown")
joined["is_change"]  = joined[l1] != joined[l2]
# Lets also measure the areas in square meters and hecter WIP
joined["area_m2"]    = joined.geometry.area
joined["area_ha"]    = joined["area_m2"] / 10_000

tm_count = (joined.groupby([l1, l2]).size()
            .reset_index(name="count")
            .pivot(index=l1, columns=l2, values="count").fillna(0).astype(int).reset_index())
tm_area  = (joined.groupby([l1, l2])["area_ha"].sum()
            .reset_index()
            .pivot(index=l1, columns=l2, values="area_ha").fillna(0).reset_index())

# Summarize the classes in each year
cnt1 = joined[l1].value_counts().rename_axis("label").reset_index(name=f"count_{year1}")
cnt2 = joined[l2].value_counts().rename_axis("label").reset_index(name=f"count_{year2}")
class_count_summary = cnt1.merge(cnt2, on="label", how="outer").fillna(0)
class_count_summary[f"diff_count_{year2}_minus_{year1}"] = class_count_summary[f"count_{year2}"] - class_count_summary[f"count_{year1}"]

area1 = joined.groupby(l1)["area_ha"].sum().rename_axis("label").reset_index(name=str(year1))
area2 = joined.groupby(l2)["area_ha"].sum().rename_axis("label").reset_index(name=str(year2))
class_area_summary = area1.merge(area2, on="label", how="outer").fillna(0)
class_area_summary[f"diff_ha_{year2}_minus_{year1}"] = class_area_summary[str(year2)] - class_area_summary[str(year1)]

total_patches = len(joined)
changed_patches = int(joined["is_change"].sum())
pct_changed = 100.0 * changed_patches / total_patches if total_patches else 0.0


change_metrics_overall = pd.DataFrame([{
    "total_patches": total_patches,
    "changed_patches": changed_patches,
    "pct_changed": pct_changed
}])

# Calculate per class metrics
labels = sorted(set(joined[l1].dropna()) | set(joined[l2].dropna()))
per_class_metrics = []
for lab in labels:
    tp = int(((joined[l1]==lab) & (joined[l2]==lab)).sum())
    fp = int(((joined[l1]!=lab) & (joined[l2]==lab)).sum())
    fn = int(((joined[l1]==lab) & (joined[l2]!=lab)).sum())
    precision = tp / (tp+fp) if (tp+fp) else 0.0
    recall = tp / (tp+fn) if (tp+fn) else 0.0
    f1 = 2*precision*recall / (precision+recall) if (precision+recall) else 0.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    per_class_metrics.append({
        "label": lab,
        f"tp_{year2}_vs_{year1}": tp,
        f"fp_{year2}_vs_{year1}": fp,
        f"fn_{year2}_vs_{year1}": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard
    })
per_class_metrics = pd.DataFrame(per_class_metrics)

net_change_count = (class_count_summary[["label", f"count_{year1}", f"count_{year2}", f"diff_count_{year2}_minus_{year1}"]]).copy()
net_change_area  = (class_area_summary[["label", str(year1), str(year2), f"diff_ha_{year2}_minus_{year1}"]]).copy()

Path(out_gpkg).unlink(missing_ok=True)
joined.to_file(out_gpkg, layer="change_tiles", driver="GPKG")
for name, df in [
    ("transition_matrix_count", tm_count),
    ("transition_matrix_area_ha", tm_area),
    ("class_count_summary", class_count_summary),
    ("class_area_summary_ha", class_area_summary),
    ("change_metrics_overall", change_metrics_overall),
    ("per_class_metrics", per_class_metrics),
    ("net_change_count", net_change_count),
    ("net_change_area_ha", net_change_area),
]:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    gpd.GeoDataFrame(df).to_file(out_gpkg, layer=name, driver="GPKG")

report_dir = Path("metrics_report_no_abandoned")
report_dir.mkdir(exist_ok=True)

# Calculate overall metrics
overall = change_metrics_overall.copy()
overall = round_df(overall, ndigits=4, cols=["pct_changed"])
overall.to_csv(report_dir / "00_overall_summary.csv", index=False)

# Calculate class counts
cc = class_count_summary.copy().sort_values("label")
for c in list(cc.columns):
    if c.startswith("count_") or c.startswith("diff_count_"):
        cc[c] = cc[c].astype("int64")
cc.to_csv(report_dir / "01_class_counts.csv", index=False)

# Class areas by hectare
ca = class_area_summary.copy().sort_values("label")
ca = round_df(ca, ndigits=2)
ca.to_csv(report_dir / "02_class_areas_ha.csv", index=False)

# Transition matrix for empirical analysis by number
tmc = tm_count.copy()
first_col = tmc.columns[0]
tmc = tmc.rename(columns={first_col: "from"})
for c in tmc.columns:
    if c != "from":
        tmc[c] = tmc[c].fillna(0).astype("int64")
tmc.to_csv(report_dir / "03_transition_matrix_counts.csv", index=False)

# Transition matrix by area
tma = tm_area.copy()
first_col_a = tma.columns[0]
tma = tma.rename(columns={first_col_a: "from"})
tma = round_df(tma, ndigits=2)
tma.to_csv(report_dir / "04_transition_matrix_area_ha.csv", index=False)

# Re-format per class metrics for csv output
pcm = per_class_metrics.copy().sort_values("label")
pcm = round_df(pcm, ndigits=4, cols=["precision", "recall", "f1", "jaccard"])
for c in ["tp_"+str(year2)+"_vs_"+str(year1), "fp_"+str(year2)+"_vs_"+str(year1), "fn_"+str(year2)+"_vs_"+str(year1)]:
    if c in pcm.columns:
        pcm[c] = pcm[c].astype("int64")
pcm.to_csv(report_dir / "05_per_class_metrics.csv", index=False)

# Calculate change by counts
ncc = net_change_count.copy().sort_values("label")
for c in list(ncc.columns):
    if c.startswith("count_") or c.startswith("diff_count_"):
        ncc[c] = ncc[c].astype("int64")
ncc.to_csv(report_dir / "06_net_change_counts.csv", index=False)

# Calculate change by area
nca = net_change_area.copy().sort_values("label")
nca = round_df(nca, ndigits=2)
nca.to_csv(report_dir / "07_net_change_area_ha.csv", index=False)