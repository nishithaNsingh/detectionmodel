import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image

# Load model and class mapping
model = load_model("plant_disease_prediction_model.h5")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_indices = {int(k): v for k, v in class_indices.items()}

# Flask setup
app = Flask(__name__)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = np.asarray(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    img = preprocess_image(img_bytes)

    preds = model.predict(img)[0]
    pred_idx = int(np.argmax(preds))
    pred_label = class_indices[pred_idx]
    confidence = round(float(np.max(preds) * 100), 2)

    return jsonify({
        "predicted_class": pred_label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
