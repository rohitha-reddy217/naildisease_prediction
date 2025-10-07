from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------------------------
# Load Trained Model
# ---------------------------
MODEL_PATH = "model/nail_disease_model.h5"
IMG_SIZE = 224
model = load_model(MODEL_PATH)

# Map numeric classes to actual disease names
CLASS_NAMES = [
    "Acral_Lentiginous_Melanoma",
    "Healthy_Nail",
    "Onychogryphosis",
    "blue_finger",
    "clubbing",
    "pitting"
]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# Helper Functions
# ---------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_disease(img_path):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)[0]

    # Only Top-1 prediction
    top_idx = np.argmax(preds)
    top_class = CLASS_NAMES[top_idx]
    confidence = round(preds[top_idx] * 100, 2)

    return top_class, confidence

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/nailhome")
def nailhome():
    return render_template("nailhome.html")

@app.route("/nailpred", methods=["GET", "POST"])
def nailpred():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Prediction
        predicted_class, confidence = predict_disease(filepath)

        return render_template(
            "nailpred.html",
            prediction=predicted_class,
            confidence=confidence,
            img_path=filepath
        )

    return render_template("nailpred.html")

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
