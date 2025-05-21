import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "crop_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Categories
categories = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
    'blast', 'brown_spot', 'dead_heart', 'downy_mildew',
    'hispa', 'normal', 'tungro'
]

def preprocess_image(image):
    """Convert image to the format expected by the model."""
    image = image.convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/')
def home():
    return render_template("index.html", prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = categories[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        return render_template("index.html", prediction=predicted_class, confidence=confidence)

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
    