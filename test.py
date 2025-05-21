import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
print(os.listdir())  # Lists all files in the current directory

# Load the trained model
model = tf.keras.models.load_model("crop_disease_model.keras")

# Define class labels (should match the order during training)
categories = ["bacterial_leaf_blight", "bacterial_leaf_streak", "bacterial_panicle_blight", "blast", "brown_spot"]

# Function to preprocess a new image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Provide the path to your test image
test_image_path = "./brown.jpeg"

# Preprocess and predict
image = preprocess_image(test_image_path)
predictions = model.predict(image)
predicted_class = np.argmax(predictions)  # Get class index
predicted_label = categories[predicted_class]  # Get class name

# Display the image with prediction
plt.imshow(cv2.imread(test_image_path)[:, :, ::-1])  # Convert BGR to RGB for display
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show()
