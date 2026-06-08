import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE     = 224
MODEL_PATH   = "skin_disease_model.h5"
CLASSES_PATH = "classes.json"

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background: #060d1a; color: #e2e8f0; }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .hero-sub { color: #64748b; font-size: 1rem; margin-top: 4px; }
  .card {
    background: #0f1f35;
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
  }
  .result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
  }
  .bar-bg {
    background: #1e3a5f;
    border-radius: 6px;
    height: 10px;
    margin: 4px 0 10px 0;
    overflow: hidden;
  }
  .bar-fill { height: 10px; border-radius: 6px; }
  .disclaimer {
    background: #0c1e35;
    border: 1px solid #1d4ed8;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #93c5fd;
  }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    st.write("Current folder:", os.getcwd())
    st.write("Model path:", MODEL_PATH)
    st.write("File exists:", os.path.exists(MODEL_PATH))

    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(MODEL_PATH)
@st.cache_resource
def load_classes():
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH) as f:
            return json.load(f)
    # Auto-detect from data/test folder
    test_dir = "data/test"
    if os.path.exists(test_dir):
        return sorted(os.listdir(test_dir))
    return []

# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, 0)

# ── Colors for classes ────────────────────────────────────────────────────────
COLORS = [
    "#38bdf8", "#818cf8", "#34d399", "#f472b6",
    "#fb923c", "#a78bfa", "#f87171", "#facc15",
    "#4ade80", "#60a5fa",
]

def get_color(i):
    return COLORS[i % len(COLORS)]

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🔬 Skin Disease Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Deep learning powered dermoscopic image analysis</p>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Medical Disclaimer:</strong> This tool is for <em>educational purposes only</em>.
  Always consult a qualified dermatologist for proper diagnosis and treatment.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

left_col, right_col = st.columns([1, 1.3], gap="large")

with left_col:
    st.markdown("### 📤 Upload Image")
    uploaded = st.file_uploader(
        "Upload a skin image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True, caption="Uploaded Image")

with right_col:
    st.markdown("### 🧬 Analysis Result")

    if not uploaded:
        st.markdown("""
        <div class="card" style="text-align:center;color:#334155;padding:3rem 1rem">
          <p style="font-size:3rem;margin:0">🩺</p>
          <p>Upload an image to start classification</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        model   = load_model()
        classes = load_classes()

        if model is None:
            st.error("Model not found! Run python train_model.py first.")
        elif not classes:
            st.error("Classes not found! Make sure data/test folder exists.")
        else:
            with st.spinner("Analysing image…"):
                tensor = preprocess(image)
                probs  = model.predict(tensor)[0]

            idx      = int(np.argmax(probs))
            top_name = classes[idx]
            top_conf = float(probs[idx]) * 100
            color    = get_color(idx)

            # Top result card
            st.markdown(f"""
            <div class="card" style="border-left:4px solid {color}">
              <p class="result-label" style="color:{color};margin-bottom:6px">
                {top_name}
              </p>
              <p style="font-size:1.3rem;font-weight:600;margin:6px 0">
                Confidence: <span style="color:{color}">{top_conf:.1f}%</span>
              </p>
            </div>
            """, unsafe_allow_html=True)

            # All classes breakdown
            st.markdown("All Classes Probability")
            sorted_idx = np.argsort(probs)[::-1]
            for i in sorted_idx:
                name = classes[i]
                conf = float(probs[i]) * 100
                c    = get_color(i)
                st.markdown(f"""
                <div style="margin-bottom:6px">
                  <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#94a3b8">
                    <span>{name}</span>
                    <span style="color:{c}">{conf:.1f}%</span>
                  </div>
                  <div class="bar-bg">
                    <div class="bar-fill" style="width:{conf}%;background:{c}"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("This app classifies skin diseases using a EfficientNetB0 deep learning model.")
    st.markdown("---")
    classes = load_classes()
    if classes:
        st.markdown("## 📋 Disease Classes")
        for i, cls in enumerate(classes):
            c = get_color(i)
            st.markdown(
                f'<span style="color:{c};font-weight:bold">● {cls}</span>',
                unsafe_allow_html=True,
            )
    st.markdown("---")
    st.markdown("Model: EfficientNetB0")
    st.markdown("Framework: TensorFlow / Keras")