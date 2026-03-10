from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from typing import Dict, List

app = FastAPI(title="Smart Monitoring API")

# ==========================================
# Load Models
# ==========================================

# YOLO Detection Model
detection_model = YOLO(
    r"D:\Graduation_Project\Model_Detection_FastApi\FastApi_M1\best.pt"
)

# CNN Accident Classification Model
accident_model = load_model(
    r"D:\Graduation_Project\Model_Accident_FastApi\accident_models\accident_model.keras"
)

# CNN(VGG16) Congestion Classification Model
congestion_model = load_model(
    r"D:\Graduation_Project\Model_Congestion_FastApi\best_congetion_model.keras"
)

models = {
    "detection": detection_model,
    "accident": accident_model,
    "congestion": congestion_model
}

# ==========================================
# WebSocket Connection Manager
# ==========================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# ==========================================
# Health Check Endpoint
# ==========================================
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# ==========================================
# WebSocket Endpoint (Real-time alerts)
# ==========================================
@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ==========================================
# Prediction Endpoint
# ==========================================
@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):

    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not available")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ======================================
    # YOLO Detection Model
    # ======================================
    if model_name == "detection":

        model = models["detection"]
        results = model(image)

        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_name = model.names[cls_id]

            detection_data = {
                "model": "detection",
                "class": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            }

            detections.append(detection_data)

            # Send real-time alert if confidence > 0.75
            if conf > 0.75:
                await manager.broadcast({
                    "type": "ALERT",
                    "model": "detection",
                    "data": detection_data
                })

        return JSONResponse({
            "success": True,
            "model": "detection",
            "count": len(detections),
            "detections": detections
        })

    # ======================================
    # CNN Accident Classification Model
    # ======================================
    elif model_name == "accident":

        model = models["accident"]

        # Preprocessing aligned with training (256 x 256)
        img = cv2.resize(image, (256, 256))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)

        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # Adjust class names according to training order
        class_names = ["Accident", "No Accident"]
        class_name = class_names[class_id]

        result_data = {
            "model": "accident",
            "class": class_name,
            "confidence": round(confidence, 4)
        }

        # Send alert if confidence > 0.75
        if confidence > 0.75:
            await manager.broadcast({
                "type": "ALERT",
                "model": "accident",
                "data": result_data
            })

        return JSONResponse({
            "success": True,
            "model": "accident",
            "prediction": result_data
        })
    
    # ======================================
    # CNN(VGG16) Congestion Classification Model
    # ======================================

    elif model_name == "congestion":

        model = models["congestion"]

        # preprocessing
        img = cv2.resize(image, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)

        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        class_names = ["Congestion", "Normal Traffic"]
        class_name = class_names[class_id]

        result_data = {
            "model": "congestion",
            "class": class_name,
            "confidence": round(confidence, 4)
        }

        # send alert
        if confidence > 0.75:
            await manager.broadcast({
                "type": "ALERT",
                "model": "congestion",
                "data": result_data
            })

        return JSONResponse({
            "success": True,
            "model": "congestion",
            "prediction": result_data
        })