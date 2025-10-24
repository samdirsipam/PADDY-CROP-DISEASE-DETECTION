import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from data_preprocessing import preprocess_image  # your function

# Load model globally (so it doesn’t reload on every request)
model = load_model("crop_disease_model.keras")

def handler(request):
    try:
        # Example: receive base64 image or file path in JSON
        data = json.loads(request.body)
        img_path = data["image_path"]  # or handle base64

        # Preprocess image
        img = preprocess_image(img_path)
        
        # Predict
        pred = model.predict(np.array([img]))
        pred_class = np.argmax(pred, axis=1)[0]

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": str(pred_class)})
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }
