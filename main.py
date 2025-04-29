import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running!"})

# Create folder to store uploads (if not exist)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
MODEL_PATH = "model/mobilenet_v2_model.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Load class labels
with open("model/dataset-details.json", "r") as f:
    class_indices = json.load(f)

# Ensure keys are strings for correct mapping
class_labels = {str(v): k for k, v in class_indices.items()}

print("Loaded class labels:", class_labels)

# Function to preprocess image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))  # Resize
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)  # Save uploaded image

    # Preprocess & predict
    image = preprocess_image(file_path)
    predictions = model.predict(image)
    predicted_index = str(np.argmax(predictions))  # Convert to string for dictionary lookup

    # Get class label
    predicted_class = class_labels.get(predicted_index, "Unknown")

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': float(np.max(predictions))  # Convert to float for JSON
    })

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
