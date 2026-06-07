import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Skin Disease Classification",
    page_icon="🩺",
    layout="centered"
)

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>

.main {
    background-color: #f4f9ff;
}

.header-box {
    background: linear-gradient(135deg, #0077ff, #00b4ff);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
}

.header-title {
    font-size: 36px;
    font-weight: bold;
}

.header-subtitle {
    font-size: 16px;
    opacity: 0.95;
}

.upload-section {
    background: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.10);
    margin-bottom: 20px;
}

.prediction-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.12);
    margin-top: 15px;
    margin-bottom: 15px;
}

.upload-note {
    text-align: center;
    color: #555;
    font-size: 15px;
}

.footer-box {
    text-align: center;
    color: #555;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# CLASS NAMES
# ==================================================
CLASS_NAMES = {
    0: "Actinic Keratoses (akiec)",
    1: "Basal Cell Carcinoma (bcc)",
    2: "Benign Keratosis-like Lesions (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic Nevi (nv)",
    6: "Vascular Lesions (vasc)"
}

# ==================================================
# DISEASE INFORMATION
# ==================================================
DISEASE_INFO = {
    "Actinic Keratoses (akiec)":
        "Actinic Keratoses are rough and scaly skin patches caused by prolonged exposure to sunlight.",

    "Basal Cell Carcinoma (bcc)":
        "Basal Cell Carcinoma is a common type of skin cancer that typically grows slowly and rarely spreads.",

    "Benign Keratosis-like Lesions (bkl)":
        "These are non-cancerous skin growths commonly observed in adults.",

    "Dermatofibroma (df)":
        "Dermatofibroma is a benign skin nodule that usually develops on the arms or legs.",

    "Melanoma (mel)":
        "Melanoma is a serious form of skin cancer that develops from pigment-producing cells.",

    "Melanocytic Nevi (nv)":
        "Melanocytic Nevi, commonly known as moles, are generally benign skin lesions.",

    "Vascular Lesions (vasc)":
        "Vascular lesions are abnormalities involving blood vessels and are often benign."
}

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/skin_disease_model.keras"
    )

model = load_model()

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="header-box">
    <div class="header-title">
        🩺 Skin Disease Classification System
    </div>
    <div class="header-subtitle">
        AI-Powered Skin Lesion Analysis using CNN and HAM10000 Dataset
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="upload-note">
Upload a dermoscopic skin image and receive an instant prediction.
</div>
""", unsafe_allow_html=True)

st.write("")

# ==================================================
# UPLOAD SECTION
# ==================================================
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# PREDICTION SECTION
# ==================================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Skin Image",
        use_container_width=True
    )

    # =================================
    # OpenCV Image Preprocessing
    # =================================

    img_array = np.array(image)

    img_array = cv2.cvtColor(
        img_array,
        cv2.COLOR_RGB2BGR
    )

    img_array = cv2.resize(
        img_array,
        (128, 128)
    )

    img_array = cv2.GaussianBlur(
        img_array,
        (3, 3),
        0
    )

    img_array = cv2.cvtColor(
        img_array,
        cv2.COLOR_BGR2RGB
    )

    img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction) * 100)

    disease_name = CLASS_NAMES[predicted_class]

    st.markdown(
        f"""
        <div class="prediction-card">
            <h3>Prediction Result</h3>
            <p><b>Predicted Disease:</b> {disease_name}</p>
            <p><b>Confidence Score:</b> {confidence:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(confidence / 100)

    st.markdown("### 📖 Disease Information")

    st.info(
        DISEASE_INFO[disease_name]
    )

    st.markdown("### 📊 Prediction Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(
            f"**{CLASS_NAMES[i]}** : {prob * 100:.2f}%"
        )
        st.progress(float(prob))

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")

st.markdown("""
<div class="footer-box">

<b>Skin Disease Classification System</b><br><br>

Technologies Used:<br>
TensorFlow • CNN • OpenCV • Streamlit • HAM10000 Dataset<br><br>

This system is intended for educational and research purposes only and should not be used as a substitute for professional medical advice.

</div>
""", unsafe_allow_html=True)