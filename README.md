# 🎓 Smart Monitoring System API – Graduation Project

## Overview
This project is a **Smart Monitoring System** designed for real-time detection, accident classification, and traffic congestion analysis using state-of-the-art Deep Learning models. It leverages **YOLO** for object detection and **CNN-based models** for accident and congestion classification, integrated with **FastAPI** for serving REST and WebSocket endpoints.

The system is capable of:
- Detecting vehicles and objects in images.
- Classifying accidents with high confidence.
- Monitoring traffic congestion.
- Broadcasting real-time alerts via WebSocket.

---

## 🏗 Project Structure

---

## ⚙ Technology Stack

- **Backend Framework:** FastAPI  
- **Real-time Communication:** WebSockets  
- **Detection Model:** YOLO (Ultralytics)  
- **Classification Models:** TensorFlow / Keras CNN models (Accident & VGG16 Congestion)  
- **Image Processing:** OpenCV & NumPy  
- **Python Version:** 3.10+ recommended  

---

## 🚀 Features

### 1. Real-Time Object Detection
- Detect vehicles, pedestrians, and other objects.
- Returns:
  - Class name
  - Bounding box coordinates
  - Confidence score
- Alerts triggered for detections with confidence > 0.75.

### 2. Accident Classification
- CNN model trained on accident dataset.
- Input: Image resized to 256×256.
- Output:  
  - Class: Accident / No Accident  
  - Confidence score
- Real-time alerts for high-confidence predictions.

### 3. Traffic Congestion Detection
- CNN VGG16 model to classify traffic conditions.
- Input: Image resized to 224×224.
- Output:  
  - Class: Congestion / Normal Traffic  
  - Confidence score
- Alerts triggered for congested traffic detected with high confidence.

### 4. WebSocket Alerts
- Push notifications in real-time.
- Active connections managed via `ConnectionManager`.
- JSON structured messages containing:
  ```json
  {
    "type": "ALERT",
    "model": "detection|accident|congestion",
    "data": { ... }
  }
