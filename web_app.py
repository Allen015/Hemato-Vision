import os 
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import io
import base64

app = Flask(__name__)
model = load_model("Blood_Cell.h5")
class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

def predict_image_class(image_path, model):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_preprocessed = preprocess_input(img_resized.reshape((1, 224, 224, 3)))
        predictions = model.predict(img_preprocessed)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        print(f"Predictions: {predictions[0]}, Chosen: {predicted_class_label}, Confidence: {confidence:.2f}")
        return predicted_class_label, img_rgb
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Error", None

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("predict.html", error="No file uploaded")
        file = request.files["file"]
        if file.filename == "":
            return render_template("predict.html", error="No file selected")
        if file:
            os.makedirs("static", exist_ok=True)
            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            predicted_class_label, img_rgb = predict_image_class(file_path, model)

            if predicted_class_label == "Error":
                return render_template("predict.html", error="Failed to process image")

            _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            img_str = base64.b64encode(img_encoded).decode('utf-8')

            return render_template("result.html", class_label=predicted_class_label, img_data=img_str)
    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)