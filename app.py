import os
os.environ["TFLITE_DISABLE_XNNPACK"] = "1"

import io
import json
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf

# --------------------------
# LOAD CLASS INDICES
# --------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_indices = {int(k): v for k, v in class_indices.items()}

# --------------------------
# LOAD TFLITE MODEL
# --------------------------
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)


# --------------------------
# IMAGE PREPROCESSING
# --------------------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)

    img_array = np.asarray(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# --------------------------
# PREDICT ROUTE
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    img = preprocess_image(img_bytes)

    # Run TFLite inference
    interpreter.set_tensor(input_details[0]["index"], img.astype("float32"))
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    pred_idx = int(np.argmax(preds))
    pred_label = class_indices[pred_idx]
    confidence = round(float(np.max(preds) * 100), 2)

    return jsonify({
        "predicted_class": pred_label,
        "confidence": confidence
    })


# --------------------------
# RUN FLASK
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
