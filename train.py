import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# =========================
# 1. LOAD METADATA
# =========================
df = pd.read_csv("dataset/HAM10000_metadata.csv")

# Map images
image_paths = {}

for folder in [
    "dataset/HAM10000_images_part_1",
    "dataset/HAM10000_images_part_2"
]:
    for img in Path(folder).glob("*.jpg"):
        image_paths[img.stem] = str(img)

df["image_path"] = df["image_id"].map(image_paths)
df = df.dropna()

# =========================
# 2. LABEL ENCODING
# =========================
classes = sorted(df["dx"].unique())
label_map = {name: idx for idx, name in enumerate(classes)}
df["label"] = df["dx"].map(label_map)

print("Class Mapping:", label_map)

# =========================
# 3. LOAD IMAGES (SAFE PIPELINE)
# =========================
IMG_SIZE = 128

def load_images(dataframe):
    images = []
    labels = []

    for _, row in dataframe.iterrows():
        img = cv2.imread(row["image_path"])
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # normalization

        images.append(img)
        labels.append(row["label"])

    return np.array(images), np.array(labels)

# =========================
# 4. TRAIN / TEST SPLIT
# =========================
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

print("Loading images... (this may take few minutes)")

X_train, y_train = load_images(train_df)
X_val, y_val = load_images(val_df)
X_test, y_test = load_images(test_df)

print("Images loaded successfully!")

# =========================
# 5. CLASS WEIGHTS
# =========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

# =========================
# 6. CNN MODEL (STABLE)
# =========================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 7. TRAIN MODEL
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    class_weight=class_weights
)

# =========================
# 8. SAVE MODEL
# =========================
model.save("model/skin_disease_model.keras")

print("\nTRAINING COMPLETE!")
print("Model saved at: model/skin_disease_model.keras")