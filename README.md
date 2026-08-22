# Autonomous Driving Real-Time Object Detection & Tracking System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange)

## Overview
This repository contains a modular computer vision pipeline designed for **Autonomous Vehicle Perception Systems**. The system utilizes state-of-the-art deep learning models (**YOLOv8**) combined with geometric spatial heuristics to achieve real-time object detection, multi-class vehicle/pedestrian tracking, and proximity-based collision risk assessment.

Developed as a core portfolio demonstration in **Pattern Recognition & Deep Learning Engineering**.

---

## System Architecture & Pipeline
1. **Input Ingestion:** Processes high-resolution dashcam video feeds or static scene images.
2. **Object Detection:** Executes multi-class inference using YOLOv8 (vehicles, pedestrians, two-wheelers).
3. **Spatial Proximity Engine:** Calculates bounding-box area ratios against frame geometry to evaluate collision thresholds.
4. **Interactive Dashboard:** Streams real-time telemetry and visual bounding overlays through a Streamlit interface.

---

## Key Features
- **Multi-Class Perception:** Real-time identification of cars, pedestrians, buses, trucks, and two-wheelers.
- **Collision Risk Heuristic:** Dynamic bounding-box spatial coverage calculation to estimate proximity and trigger automated collision alerts.
- **Interactive Control Panel:** Streamlit web UI featuring adjustable confidence thresholds and live telemetry statistics.
- **Modular Design:** Clear separation between inference logic (`src/detector.py`) and application interface (`app.py`).

---

## Technologies & Frameworks
- **Language:** Python
- **Computer Vision:** OpenCV, PyTorch, Ultralytics YOLOv8
- **Frontend / Dashboard:** Streamlit
- **Data Handling:** NumPy, Pandas, PIL
