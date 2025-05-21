import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define constants
DATASET_PATH = "./dataset"  # CHANGE THIS TO YOUR DATASET PATH
IMG_SIZE = 128  # Resize all images to 128x128
TEST_SPLIT = 0.2  # 80% training, 20% testing

# Step 1: Define Updated Class Labels
categories = [
    "bacterial_leaf_blight", "bacterial_leaf_streak", "bacterial_panicle_blight",
    "blast", "brown_spot", "dead_heart", "downy_mildew",
    "hispa", "normal", "tungro"
]
label_map = {category: i for i, category in enumerate(categories)}
print("Class Labels:", label_map)

# Step 2: Load Images & Assign Labels
def load_and_preprocess_images(directory, img_size):
    data, labels = [], []
    
    for category in categories:
        path = os.path.join(directory, category)
        label = label_map[category]
        
        if not os.path.exists(path):
            print(f"Warning: Directory {path} not found, skipping...")
            continue

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}, skipping...")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            labels.append(label)

    return np.array(data), np.array(labels)

# Load dataset
X, y = load_and_preprocess_images(DATASET_PATH, IMG_SIZE)

# Step 3: Normalize the Images (Scale pixel values to [0,1])
X = X / 255.0  

# Step 4: Convert Labels to One-Hot Encoding
y = to_categorical(y, num_classes=len(categories))  

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42, stratify=y)

# Print dataset info
print("\nDataset Loaded Successfully!")
print("Training Data Shape:", X_train.shape, y_train.shape)
print("Testing Data Shape:", X_test.shape, y_test.shape)

# Step 6: Show Sample Images from Different Classes
plt.figure(figsize=(12, 6))

unique_classes_shown = set()
count = 0

for i in range(len(X_train)):
    label_index = np.argmax(y_train[i])
    label_name = categories[label_index]
    
    if label_name not in unique_classes_shown:
        plt.subplot(2, 5, count + 1)
        plt.imshow(X_train[i])
        plt.title(f"Label: {label_name}")
        plt.axis("off")
        unique_classes_shown.add(label_name)
        count += 1
    
    if count == len(categories):
        break

plt.tight_layout()
plt.show()