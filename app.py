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
model = None

# Categories
categories = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
    'blast', 'brown_spot', 'dead_heart', 'downy_mildew',
    'hispa', 'normal', 'tungro'
]

# Disease-specific suggestions
disease_suggestions = {
    'bacterial_leaf_blight': "Remove infected leaves, use copper-based fungicides, and ensure proper spacing.",
    'bacterial_leaf_streak': "Avoid overhead irrigation and apply recommended bactericides.",
    'bacterial_panicle_blight': "Use resistant varieties and maintain proper field sanitation.",
    'blast': "Apply fungicides like tricyclazole, maintain water management, and rotate crops.",
    'brown_spot': "Use balanced fertilizers and remove infected debris.",
    'dead_heart': "Ensure proper nitrogen management and control stem borer insects.",
    'downy_mildew': "Apply systemic fungicides and ensure proper drainage.",
    'hispa': "Monitor for insects and use appropriate insecticides.",
    'normal': "No disease detected. Continue good agricultural practices.",
    'tungro': "Control leafhopper insects and remove infected plants immediately."
}


def get_model():
    """Load the model only once when needed."""
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

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

        model_instance = get_model()
        prediction = model_instance.predict(processed_image)
        
        predicted_class = categories[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        suggestion = disease_suggestions.get(predicted_class, "No suggestions available.")
        return render_template("index.html", prediction=predicted_class, confidence=confidence, suggestion=suggestion)

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
    