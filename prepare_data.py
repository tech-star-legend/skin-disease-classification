import pandas as pd
from pathlib import Path

# Load metadata
df = pd.read_csv("dataset/HAM10000_metadata.csv")

# Collect all image paths
image_paths = {}

for folder in [
    "dataset/HAM10000_images_part_1",
    "dataset/HAM10000_images_part_2"
]:
    for img_path in Path(folder).glob("*.jpg"):
        image_paths[img_path.stem] = str(img_path)

# Match image_id with image file path
df["image_path"] = df["image_id"].map(image_paths)

print("Dataset Shape:", df.shape)

print("\nMissing Images:")
print(df["image_path"].isnull().sum())

print("\nDisease Distribution:")
print(df["dx"].value_counts())

print("\nSample Records:")
print(df[["image_id", "dx", "image_path"]].head())