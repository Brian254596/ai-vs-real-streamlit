# ================================
# app.py
# ================================

import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import numpy as np

# -------------------------------
# 1️⃣ Load Model & Labels (once)
# -------------------------------

@st.experimental_singleton  # ensures it's loaded only once per session
def load_model_and_labels():
    # Load Keras model safely
    model = load_model("ai_vs_real_model.keras", compile=False)

    # Load class labels
    with open("labels.pkl", "rb") as f:
        class_names = pickle.load(f)

    return model, class_names

model, class_names = load_model_and_labels()

# -------------------------------
# 2️⃣ Streamlit App Layout
# -------------------------------

st.set_page_config(page_title="Real vs AI Image Classifier", layout="centered")
st.title("🖼️ Real vs AI Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # 3️⃣ Preprocess Image
    # -------------------------------
    img = image.resize((224, 224))  # same size used in training
    img_array = np.array(img) / 255.0  # normalize to 0-1
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # -------------------------------
    # 4️⃣ Make Prediction
    # -------------------------------
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][pred_index]

    st.write(f"Prediction: **{class_names[pred_index]}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")