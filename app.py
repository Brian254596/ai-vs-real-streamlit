# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

# ---- 1️⃣ Load Model & Labels ----
@st.cache_resource
def load_model_and_labels():
    model = load_model("ai_vs_real_model.keras")
    with open("labels.pkl", "rb") as f:
        class_names = pickle.load(f)
    return model, class_names

model, class_names = load_model_and_labels()

# ---- 2️⃣ Streamlit App Layout ----
st.set_page_config(page_title="Real vs AI Image Classifier", layout="centered")
st.title("🖼️ Real vs AI Image Classifier")
st.write("Upload one or multiple images and the model will predict whether they are Real or AI-generated.")

uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", width=700)

        # Preprocess image
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalize

        # Predict (for 1-neuron sigmoid binary output)
        prediction = model.predict(img_array)[0][0]  # scalar probability for "AI"
        if prediction >= 0.5:
            class_name = class_names[1]  # AI
            confidence = prediction
        else:
            class_name = class_names[0]  # Real
            confidence = 1 - prediction

        # Save result
        results.append({"Image": uploaded_file.name, "Predicted Class": class_name, "Confidence": f"{confidence*100:.2f}%"})

    # Display all results in a table
    st.write("### Prediction Results")
    st.table(results)