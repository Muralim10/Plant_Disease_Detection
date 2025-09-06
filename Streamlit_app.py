# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Image Classifier", layout="centered")

st.title("My Image Classifier")
st.write("Upload an image and I'll tell you what I think!")

# ---- 1) Load model (cached so Streamlit doesn't reload it all the time) ----
@st.cache_resource
def load_my_model(path):
    model = tf.keras.models.load_model(path)
    return model

MODEL_PATH = "model.h5"   # <- put your model file name here
model = load_my_model(MODEL_PATH)

# ---- 2) Figure out expected input size from the model ----
try:
    input_shape = model.input_shape[1:3]   # e.g. (224, 224)
    input_shape = (int(input_shape[0]), int(input_shape[1]))
except Exception:
    input_shape = (224, 224)  # fallback

# ---- 3) Replace these with your real class names ----
class_names = ["class_0", "class_1", "class_2"]  # <-- edit this!

# ---- 4) Helpers ----
def preprocess_image(img: Image.Image, target_size):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, 3)
    return arr

def predict(image: Image.Image):
    x = preprocess_image(image, input_shape)
    preds = model.predict(x)[0]   # model outputs: e.g. logits or probabilities
    # If outputs are logits, convert to probabilities:
    try:
        probs = tf.nn.softmax(preds).numpy()
    except Exception:
        probs = preds
    return probs

# ---- 5) UI: upload file ----
uploaded = st.file_uploader("Choose an image (jpg, png)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)

    with st.spinner("Running the model..."):
        probs = predict(img)

    # If single value output (regression), just show it:
    if probs.ndim == 0 or probs.size == 1:
        st.write("Model output:", float(probs))
    else:
        top_idx = int(np.argmax(probs))
        st.markdown(f"**Prediction:** `{class_names[top_idx]}`")
        st.write(f"Confidence: {probs[top_idx]*100:.2f}%")

        # show top 3
        topk = np.argsort(probs)[::-1][:3]
        st.write("Top predictions:")
        for i in topk:
            st.write(f"- {class_names[i]}: {probs[i]*100:.2f}%")
