import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.detector import PerceptionEngine

st.set_page_config(page_title="Autonomous Driving Perception", layout="wide")

st.title("🚗 Autonomous Vehicle Perception System")
st.sidebar.header("Control Panel")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35, 0.05)

@st.cache_resource
def load_engine():
    return PerceptionEngine()

engine = load_engine()

uploaded_file = st.sidebar.file_uploader("Upload Dashcam Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Dashcam Input")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
    processed_img, alert_triggered, detected_count = engine.process_frame(image, conf_threshold)
    
    with col2:
        st.subheader("Perception Pipeline Output")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
        
    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Objects Detected", detected_count)
    m2.metric("Proximity Warning", "CRITICAL RISK" if alert_triggered else "CLEAR", delta_color="inverse")
else:
    st.info("Please upload a dashcam frame using the sidebar to run perception inference.")
