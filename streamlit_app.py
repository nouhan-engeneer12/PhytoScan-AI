import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

st.set_page_config(
    page_title="PhytoScan AI",
    page_icon="🌿",
    layout="centered"
)

MODEL_PATH = "PhytoScan_AI.keras"

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]

DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "name": "Bacterial Spot",
        "symptoms": "Small dark spots that may develop yellow edges.",
        "advice": "Avoid wetting leaves and seek local agricultural advice."
    },
    "Tomato___Early_blight": {
        "name": "Early Blight",
        "symptoms": "Brown leaf spots may show concentric rings and yellowing.",
        "advice": "Remove badly affected leaves and seek agricultural guidance."
    },
    "Tomato___Late_blight": {
        "name": "Late Blight",
        "symptoms": "Dark, spreading lesions may appear on leaves.",
        "advice": "Separate affected plants if possible and seek prompt advice."
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato Leaf",
        "symptoms": "The model classified the image as healthy.",
        "advice": "Continue monitoring the plant for changes."
    }
}

st.title("🌿 PhytoScan AI")
st.subheader("AI-Based Plant Disease Detection")
st.write("Upload a tomato leaf image to get a predicted class and general guidance.")

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Upload PhytoScan_AI.keras "
        "to the same GitHub repository as this app."
    )
    st.stop()

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    if st.button("Analyze Leaf"):
        with st.spinner("Analyzing image..."):
            resized = image.resize((128, 128))
            array = np.array(resized, dtype=np.float32) / 255.0
            array = np.expand_dims(array, axis=0)

            model = get_model()
            predictions = np.asarray(model.predict(array))[0]

            index = int(np.argmax(predictions))
            class_name = CLASS_NAMES[index]
            confidence = float(predictions[index]) * 100
            info = DISEASE_INFO[class_name]

        st.header("Analysis Result")
        st.write("**Predicted class:**", info["name"])
        st.write(f"**Model confidence:** {confidence:.1f}%")

        st.subheader("General symptoms")
        st.write(info["symptoms"])

        st.subheader("Suggested next step")
        st.write(info["advice"])

        st.caption(
            "This is an AI prediction, not a confirmed diagnosis. "
            "The model supports tomato leaves and may make mistakes."
        )
