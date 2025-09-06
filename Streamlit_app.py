# app.py - Streamlit app for Keras/TensorFlow model
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(page_title="Image Classifier", layout="centered")

# ---------------- Settings ----------------
MODEL_PATH = "models/my_model.h5"     # change path if needed
CLASS_JSON = "models/class_names.json"  # file with class names in training order
IMAGE_SIZE = (224, 224)               # change to your training input size
PREPROCESS = "mobilenet_v2"           # set according to training: "mobilenet_v2", "efficientnet", "resnet50", or "none"

# ---------------- Load Model & Metadata ----------------
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_class_names(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return [f"class_{i}" for i in range(100)]  # fallback dummy names

model = load_model(MODEL_PATH)
class_names = load_class_names(CLASS_JSON)

# ---------------- Preprocessing ----------------
def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img).astype(np.float32)

    if PREPROCESS == "mobilenet_v2":
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    elif PREPROCESS.startswith("efficientnet"):
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    elif PREPROCESS == "resnet50":
        arr = tf.keras.applications.resnet50.preprocess_input(arr)
    else:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)

# ---------------- Prediction ----------------
def predict_topk(pil_img, top_k=3):
    x = preprocess_image(pil_img)
    preds = model.predict(x)[0]
    probs = tf.nn.softmax(preds).numpy()
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top_idx]

# ---------------- UI ----------------
st.title("📷 Keras Image Classifier")
st.write("Upload an image and let the model predict its class.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
top_k = st.sidebar.slider("Show top K predictions", 1, min(10, len(class_names)), 3)

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Input Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            results = predict_topk(img, top_k=top_k)

        st.success("Prediction complete!")

        # Show results
        st.write("### Predictions")
        for name, prob in results:
            st.write(f"- **{name}**: {prob*100:.2f}%")

        # Show bar chart
        labels = [name for name, _ in results]
        values = [prob*100 for _, prob in results]
        fig, ax = plt.subplots()
        ax.barh(labels[::-1], values[::-1])
        ax.set_xlabel("Probability (%)")
        st.pyplot(fig)
