# app.py (plant-disease-ready)
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import pandas as pd

st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image and the model will tell you the disease (or if it's healthy).")

# ---------- EDIT THESE ----------
MODEL_PATH = "model.h5"        # <-- your model file or saved_model folder
LABELS_PATH = "labels.txt"     # <-- optional file with one label per line (train order)
USE_IMAGENET_PREPROCESS = False  # <-- True if you used tf.keras.applications preprocessing
IMAGENET_MODEL = "mobilenet_v2"  # e.g. "mobilenet_v2", "efficientnet", "resnet50" (only used if above True)
# --------------------------------

@st.cache_resource
def load_model(path):
    # If you used custom layers, add custom_objects={'MyLayer': MyLayer}
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)

# Determine input shape (H, W)
try:
    input_shape = model.input_shape[1:3]
    input_shape = (int(input_shape[0]), int(input_shape[1]))
except Exception:
    input_shape = (224, 224)

# Load labels if provided, otherwise create placeholders
def load_labels(path, n_classes):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
        # If labels count doesn't match model output, warn and fallback
        if n_classes is not None and len(labels) != n_classes:
            st.warning(f"labels.txt has {len(labels)} entries but model predicts {n_classes}.")
        return labels
    else:
        if n_classes is None:
            return ["class_0"]
        return [f"class_{i}" for i in range(n_classes)]

# Get number of output classes if possible
try:
    n_out = model.output_shape[-1]
except Exception:
    n_out = None

class_names = load_labels(LABELS_PATH, n_out)

# Preprocess helper (match this to what you used during training)
def preprocess_image(img: Image.Image, target_size):
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img).astype(np.float32)
    if USE_IMAGENET_PREPROCESS:
        if IMAGENET_MODEL == "mobilenet_v2":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        elif IMAGENET_MODEL.startswith("efficientnet"):
            from tensorflow.keras.applications.efficientnet import preprocess_input
        elif IMAGENET_MODEL == "resnet50":
            from tensorflow.keras.applications.resnet import preprocess_input
        else:
            # fallback; user may need to change this import
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        arr = preprocess_input(arr)
    else:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(img: Image.Image):
    x = preprocess_image(img, input_shape)
    preds = model.predict(x)
    preds = np.squeeze(preds)  # remove batch dim
    # Single output (regression / binary)
    if np.ndim(preds) == 0 or np.size(preds) == 1:
        # If it's a single logit, assume sigmoid may be needed:
        val = float(preds)
        if val < 0 or val > 1:
            try:
                val = float(tf.sigmoid(val).numpy())
            except Exception:
                pass
        return {"single_value": val}
    # Multi-class output
    else:
        # If outputs look like logits (not summing to 1), apply softmax:
        if preds.dtype.kind in 'f' and not np.isclose(preds.sum(), 1.0):
            probs = tf.nn.softmax(preds).numpy()
        else:
            probs = np.array(preds)
        return {"probs": probs}

# ---------- UI ----------
uploaded = st.file_uploader("Choose an image (jpg, png)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)

    with st.spinner("Running model..."):
        out = predict_image(img)

    if "single_value" in out:
        prob = out["single_value"]
        st.write(f"Model output (interpreted probability): {prob*100:.2f}%")
        st.write("If this is a binary classifier, choose a threshold to decide disease/healthy:")
        thresh = st.slider("Threshold", 0.0, 1.0, 0.5)
        label = "diseased" if prob >= thresh else "healthy"
        st.markdown(f"**Prediction:** `{label}` (threshold {thresh})")
    else:
        probs = out["probs"]
        # safety: ensure length matches labels
        if len(probs) != len(class_names):
            st.error(f"Model output length ({len(probs)}) != label count ({len(class_names)}). Update labels.txt or check model.")
        else:
            df = pd.DataFrame({
                "label": class_names,
                "probability (%)": (probs * 100)
            }).sort_values("probability (%)", ascending=False)
            st.markdown(f"**Top prediction:** `{df.iloc[0]['label']}` — {df.iloc[0]['probability (%)']:.2f}%")
            st.table(df.reset_index(drop=True))
            # bar chart of top 10
            st.bar_chart(df.set_index("label")["probability (%)"].head(10))

# Optional: camera input
st.write("---")
st.write("Tip: You can also take a photo with your camera (useful on phones).")
cam = st.camera_input("Or snap a leaf")
if cam is not None:
    img2 = Image.open(cam)
    st.image(img2, caption="Camera image", use_column_width=True)
    with st.spinner("Running model on camera image..."):
        out2 = predict_image(img2)
    if "single_value" in out2:
        st.write(f"Probability: {out2['single_value']*100:.2f}%")
    else:
        probs2 = out2["probs"]
        if len(probs2) == len(class_names):
            df2 = pd.DataFrame({"label": class_names, "probability (%)": (probs2*100)}).sort_values("probability (%)", ascending=False)
            st.markdown(f"**Top:** `{df2.iloc[0]['label']}` — {df2.iloc[0]['probability (%)']:.2f}%")
            st.table(df2.reset_index(drop=True))
        else:
            st.error("Mismatch between model outputs and labels.")

st.write("---")
st.caption("Make sure `labels.txt` matches the class order used during training (one label per line).")
