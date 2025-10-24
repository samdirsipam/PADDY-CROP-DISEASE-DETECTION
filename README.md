# AI-Driven Paddy Crop Disease Detection

[Live Demo](https://plantspy.onrender.com)

---

Overview
This project is an AI-powered web application that detects diseases in paddy crops using images of leaves. It provides "confidence scores" and "disease-specific suggestions" to help farmers take corrective actions.  

The system uses a "deep learning model" trained on paddy crop disease data, served via a Flask web application with a modern frontend.


Features
- Upload an image of a paddy leaf.
- Detect diseases from **10 categories**:
  - bacterial_leaf_blight
  - bacterial_leaf_streak
  - bacterial_panicle_blight
  - blast
  - brown_spot
  - dead_heart
  - downy_mildew
  - hispa
  - normal
  - tungro
- Show "confidence score" for predictions.
- Display "actionable suggestions" for each disease.
- Modern, responsive, and interactive frontend.
- Automatically clears previous predictions when uploading a new image.


Technologies Used
- Backend Python, Flask  
- Frontend HTML, CSS, JavaScript  
- Machine Learning TensorFlow, Keras  
- Deployment Render


Project Structure

PADDY-CROP-DISEASE-SETECTION/
├── app.py # Main Flask application
├── crop_disease_model.keras # Trained TensorFlow model
├── data_preprocessing.py # Data preprocessing script
├── train_crop_disease.py # Model training script
├── test.py # Test scripts
├── templates/
│ └── index.html (includes HMTL, CSS, JS)
├── requirements.txt # Python dependencies




Setup & Installation

1. Clone the repository

git clone https://github.com/samdirsipam/PADDY-CROP-DISEASE-DETECTION.git
cd PADDY-CROP-DISEASE-DETECTION


2. Create a virtual environment:

python -m venv venv
# Activate environment
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

3. pip install -r requirements.txt
4. python app.py



Note:
1. Ensure crop_disease_model.keras is in the project root before running.
2. Currently supports image uploads only (JPEG/PNG).
3. Suggestions are based on common agricultural practices for paddy crops.
4. Confidence score is rounded to 3 decimal places.
