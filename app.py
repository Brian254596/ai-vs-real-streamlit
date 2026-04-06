import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import numpy as np

# Load model and labels
@st.cache_resource
def load_model_and_labels():
    model = load_model("ai_vs_real_model.h5", compile=False)
    with open("labels.pkl", "rb") as f:
        class_names = pickle.load(f)
    return model, class_names

model, class_names = load_model_and_labels()

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    arr = np.array(image)/255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# Streamlit UI
st.title("Real vs AI Image Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    input_arr = preprocess_image(image)
    prediction = model.predict(input_arr)[0][0]

    # Show full bold message
    if prediction > 0.5:
        st.markdown("### **THE IMAGE IS AI GENERATED**")
    else:
        st.markdown("### **THE IMAGE IS REAL**")