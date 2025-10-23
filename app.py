import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model
MODEL_PATH = "models/traffic_sign_final.h5"

st.set_page_config(page_title="Traffic Sign Recognition", layout="centered")

st.title("🚦 Traffic Sign Recognition System")
st.write("Upload an image of a traffic sign to predict its category.")

# Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Define class names (you can modify based on your dataset)
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    14: 'Stop',
    17: 'No entry',
    18: 'General caution'
}

# Upload section
uploaded_file = st.file_uploader("📸 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Traffic Sign", use_container_width=True)

    # Preprocess image for model
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction button
    if st.button("🔍 Predict"):
        with st.spinner("Predicting..."):
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            label = classes.get(predicted_class, "Unknown Sign")

        st.success(f"✅ Predicted Sign: **{label}**")
else:
    st.info("👆 Please upload an image file (PNG/JPG).")

