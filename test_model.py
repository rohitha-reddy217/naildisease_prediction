import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = "model/nail_disease_model.h5"
TEST_DIR = "Dataset/test"
IMG_SIZE = 224

# Map numeric classes to actual disease names
CLASS_NAMES = [
    "Acral_Lentiginous_Melanoma",  # Class_0
    "Healthy_Nail",                 # Class_1
    "Onychogryphosis",              # Class_2
    "blue_finger",                  # Class_3
    "clubbing",                     # Class_4
    "pitting"                       # Class_5
]

# ---------------------------
# Load Model
# ---------------------------
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# ---------------------------
# Prediction Function
# ---------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    preds = model.predict(img_array)[0]
    predicted_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_index]
    probabilities = {CLASS_NAMES[i]: round(preds[i] * 100, 2) for i in range(len(CLASS_NAMES))}

    return predicted_class, probabilities

# ---------------------------
# Evaluate on Test Folder
# ---------------------------
correct = 0
total = 0

for class_folder in os.listdir(TEST_DIR):
    class_path = os.path.join(TEST_DIR, class_folder)
    if not os.path.isdir(class_path):
        continue

    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        predicted_class, probs = predict_image(img_path)
        total += 1
        if predicted_class == class_folder:
            correct += 1
        print(f"[{total}] Image: {img_file} | True: {class_folder} | Predicted: {predicted_class}")

accuracy = correct / total if total > 0 else 0
print(f"\n✅ Test Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
