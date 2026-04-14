import streamlit as st
import joblib
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Brain Tumor Detection", layout="centered", page_icon="🧠")

st.title("🧠 Brain Tumor Detection")

st.write("Upload an MRI brain scan to check for the presence of a tumor.")

@st.cache_resource
def load_classification_model():
    model_path = os.path.join('models', 'tumor_model.pkl')
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_classification_model()

if model is None:
    st.error("Model not found. Please train the model and ensure it's saved in the 'models' directory.")
else:
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded MRI.', width=300)
        st.write("")
        st.write("Processing...")
        
        # Preprocess the image (needs to match training exactly)
        img = image.convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img, dtype="float32") / 255.0
        
        # Flatten for scikit-learn
        img_array_flat = img_array.reshape(1, -1)
        
        # Predict probabilities
        prediction_prob = model.predict_proba(img_array_flat)[0]
        tumor_prob = prediction_prob[1]
        
        # Interpret
        if tumor_prob > 0.5:
            st.error(f"⚠️ **Tumor Detected** (Confidence: {tumor_prob * 100:.2f}%)")
        else:
            st.success(f"✅ **No Tumor Detected** (Confidence: {(1 - tumor_prob) * 100:.2f}%)")
