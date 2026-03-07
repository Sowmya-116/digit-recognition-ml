from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import io
from PIL import Image
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model from the environment variable
model_path = os.environ.get("MODEL_PATH")
if not model_path:
    raise ValueError("MODEL_PATH environment variable not set.")
model = load_model(model_path)

def preprocess_image(image):
    image = image.convert("L").resize((28, 28))
    image = 255 - np.array(image)
    image = image / 255.0
    image = image.reshape(1, 28, 28)
    return image

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data["image"].split(",")[1]

    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    processed = preprocess_image(image)

    prediction = model.predict(processed)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    return jsonify({
        "digit": digit,
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)