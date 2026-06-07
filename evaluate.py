import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# Load metadata
df = pd.read_csv("dataset/HAM10000_metadata.csv")

# Image paths
image_paths = {}

for folder in [
    "dataset/HAM10000_images_part_1",
    "dataset/HAM10000_images_part_2"
]:
    for img in Path(folder).glob("*.jpg"):
        image_paths[img.stem] = str(img)

df["image_path"] = df["image_id"].map(image_paths)

# Labels
classes = sorted(df["dx"].unique())
label_map = {name: idx for idx, name in enumerate(classes)}

df["label"] = df["dx"].map(label_map)

# Split exactly like training
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

# Load test images
X_test = []
y_test = []

for _, row in test_df.iterrows():
    img = cv2.imread(row["image_path"])
    img = cv2.resize(img, (128,128))
    img = img / 255.0

    X_test.append(img)
    y_test.append(row["label"])

X_test = np.array(X_test)
y_test = np.array(y_test)

# Load model
model = load_model("model/skin_disease_model.keras")

# Predict
predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))