import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Example Training History
accuracy = [0.11, 0.08, 0.10, 0.18, 0.35, 0.42, 0.42, 0.48, 0.46, 0.49]
val_accuracy = [0.03, 0.06, 0.05, 0.20, 0.49, 0.51, 0.54, 0.55, 0.47, 0.60]

loss = [1.96, 1.94, 1.90, 1.87, 1.63, 1.54, 1.49, 1.39, 1.39, 1.30]
val_loss = [1.94, 1.91, 1.96, 1.88, 1.24, 1.34, 1.19, 1.14, 1.22, 1.01]

# ==========================
# Accuracy Graph
# ==========================
plt.figure(figsize=(8,5))
plt.plot(accuracy, label="Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# Loss Graph
# ==========================
plt.figure(figsize=(8,5))
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# Confusion Matrix
# ==========================

cm = np.array([
    [14,10,0,1,3,4,0],
    [9,22,2,3,3,13,0],
    [10,24,8,18,20,30,0],
    [4,3,1,2,0,1,0],
    [5,2,5,2,61,35,2],
    [19,19,6,8,106,508,5],
    [0,0,0,0,0,1,13]
])

class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()