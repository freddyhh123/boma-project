import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("patches/patches_metadata.csv")

df["label"].value_counts().sort_index().plot(kind='bar', color='lightcoral',zorder=3)
plt.title("Label Distribution in Patch Dataset")
plt.xlabel("Label (0 = No settlement, 1 = Abandoned, 2 = Active)")
plt.ylabel("Number of Patches")
plt.xticks(rotation=0)
plt.grid(axis='y', zorder=0)
plt.tight_layout()
plt.show()
