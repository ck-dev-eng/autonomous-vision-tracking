import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from src.detector import AutonomousDetector

st.set_page_config(
    page_title="Autonomous Perception Suite",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Autonomous Driving Perception & Object Tracking System")
st.markdown("""
**Real-Time Pattern Recognition & Risk Analysis Dashboard**
This application utilizes **YOLOv8** and **OpenCV** to detect vehicles, pedestrians, and obstacles in real-time, estimating proximity and collision risk for autonomous vehicles.
""")

st.sidebar.header("System Configurations")
conf_thresh = st.sidebar.slider("Detection Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
model_size = st.sidebar.selectbox("YOLOv8 Model Size", ["yolov8n.pt", "yolov8s.pt"])

@st.cache_resource
def load_detector(model_name):
    return AutonomousDetector(model_version=model_name)

detector = load_detector(model_size)

source_mode = st.sidebar.radio("Select Input Source", ["Image Analysis", "Video Stream"])

if source_mode == "Image Analysis":
    uploaded_file = st.file_uploader("Upload Driving Scene Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        processed_frame, counts, danger = detector.process_frame(frame, conf_threshold=conf_thresh)
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(processed_frame_rgb, caption="Processed Autonomous Vision Feed", use_column_width=True)
        with col2:
            st.subheader("Real-Time Telemetry")
            if danger:
                st.error("🚨 CRITICAL WARNING: Proximity Limit Exceeded!")
            else:
                st.success("✅ Path Clear - Normal Navigation")
            
            st.metric("Detected Cars", counts["car"])
            st.metric("Detected Pedestrians", counts["person"])
            st.metric("Trucks / Buses", counts["truck/bus"])
            st.metric("Motorcycles / Bicycles", counts["two-wheeler"])

elif source_mode == "Video Stream":
    uploaded_video = st.file_uploader("Upload Dashcam Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        st_frame = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, counts, danger = detector.process_frame(frame, conf_threshold=conf_thresh)
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            st_frame.image(processed_frame_rgb, channels="RGB", use_column_width=True)

        cap.release()
