import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import numpy as np

# ------------------------------
# Load model and labels
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    # Load the trained model (HDF5 format)
    model = load_model("ai_vs_real_model.h5", compile=False)
    
    # Load class labels
    with open("labels.pkl", "rb") as f:
        class_names = pickle.load(f)
    
    return model, class_names

model, class_names = load_model_and_labels()

# ------------------------------
# Image preprocessing
# ------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))           # Resize to model input
    arr = np.array(image)/255.0                 # Normalize pixels
    arr = np.expand_dims(arr, axis=0)          # Add batch dimension
    return arr

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Real vs AI Image Classifier")

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Open and ensure RGB
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    input_arr = preprocess_image(image)
    prediction = model.predict(input_arr)[0][0]
    
    # Determine label
    label = class_names[1] if prediction > 0.5 else class_names[0]
    st.write(f"Prediction: **{label}** ({prediction*100:.2f}%)")