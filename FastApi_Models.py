from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from typing import List

app = FastAPI(title="Smart Monitoring API")

# ==========================================
# Load Models
# ==========================================
detection_model = YOLO("best.pt")
accident_model = load_model("accident_model.keras")
congestion_model = load_model("best_congetion_model.keras")

models = {
    "detection": detection_model,
    "accident": accident_model,
    "congestion": congestion_model
}

# ==========================================
# CAMERAS (100 LOCATIONS IN EGYPT)
# ==========================================
cameras = {
    1: "Nasr City",
    2: "Maadi",
    3: "Dokki",
    4: "Giza",
    5: "Heliopolis",
    6: "6th October City",
    7: "Downtown Cairo",
    8: "Alexandria Corniche",
    9: "Ring Road Cairo",
    10: "New Cairo",
    11: "Shubra El Kheima",
    12: "Imbaba",
    13: "Faisal Street",
    14: "Mohandessin",
    15: "Zamalek",
    16: "Garden City",
    17: "Helwan",
    18: "Banha",
    19: "Tanta",
    20: "Mansoura",
    21: "Zagazig",
    22: "Ismailia",
    23: "Suez",
    24: "Port Said",
    25: "Damietta",
    26: "Fayoum",
    27: "Beni Suef",
    28: "Minya",
    29: "Assiut",
    30: "Sohag",
    31: "Qena",
    32: "Luxor",
    33: "Aswan",
    34: "Hurghada",
    35: "Sharm El Sheikh",
    36: "North Coast",
    37: "El Alamein",
    38: "Smart Village",
    39: "Sheikh Zayed",
    40: "El Rehab City",
    41: "Madinty",
    42: "New Capital",
    43: "Obour City",
    44: "El Shorouk",
    45: "10th of Ramadan",
    46: "15 May City",
    47: "Badr City",
    48: "Ain Shams",
    49: "Mataria",
    50: "Rod El Farag",
    51: "Boulaq",
    52: "Sayeda Zeinab",
    53: "Bab El Shaaria",
    54: "Abdeen",
    55: "Kasr El Nile",
    56: "Salah Salem",
    57: "Autostrad Road",
    58: "26th July Corridor",
    59: "Corniche El Nile",
    60: "El Haram Street",
    61: "Giza Pyramids Area",
    62: "Heliopolis Square",
    63: "Nasr City 1st District",
    64: "Nasr City 2nd District",
    65: "New Cairo 5th Settlement",
    66: "Tagamoa",
    67: "Katameya Heights",
    68: "Cairo Airport Road",
    69: "Alex Desert Road",
    70: "Suez Road",
    71: "Ismailia Road",
    72: "Fayoum Road",
    73: "Wadi El Natroun Road",
    74: "Borg El Arab",
    75: "Smouha Alexandria",
    76: "Agami",
    77: "El Marg",
    78: "Shubra Masr",
    79: "Helwan University Area",
    80: "Cairo Stadium Area",
    81: "New Heliopolis",
    82: "El Waily",
    83: "Old Cairo",
    84: "Fustat",
    85: "Industrial Zone 6th October",
    86: "Media Production City",
    87: "Smart Village Entrance",
    88: "Cairo Ring Road North",
    89: "Cairo Ring Road South",
    90: "Red Sea Highway",
    91: "Upper Egypt Highway",
    92: "Western Desert Road",
    93: "North Sinai Route",
    94: "South Sinai Mountains",
    95: "El Nasr Road",
    96: "Borg El Arab Airport Road",
    97: "Cairo-Alex Desert Road",
    98: "El Marg Bridge",
    99: "Cairo Metro Surroundings",
    100: "Giza Corridor"
}

# ==========================================
# Preprocessing Functions
# ==========================================
def preprocess_detection(image):
    return image

def preprocess_accident(image):
    img = cv2.resize(image, (256, 256))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_congestion(image):
    img = cv2.resize(image, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

preprocessing_functions = {
    "detection": preprocess_detection,
    "accident": preprocess_accident,
    "congestion": preprocess_congestion
}

# ==========================================
# WebSocket Manager
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
# Health Check
# ==========================================
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# ==========================================
# WebSocket Endpoint
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
async def predict(
    model_name: str,
    file: UploadFile = File(...),
    camera_id: str = Form(...)
):

    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not available")

    # Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Preprocessing
    processed_image = preprocessing_functions[model_name](image)

    # Camera location
    location = cameras.get(int(camera_id), "Unknown Location")

    # ==========================================
    # YOLO Detection
    # ==========================================
    if model_name == "detection":

        model = models["detection"]
        results = model(processed_image)

        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_name = model.names[cls_id]

            detection_data = {
                "camera_id": camera_id,
                "location": location,
                "model": "detection",
                "class": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            }

            detections.append(detection_data)

            if conf > 0.75:
                await manager.broadcast({
                    "type": "ALERT",
                    "camera_id": camera_id,
                    "location": location,
                    "model": "detection",
                    "data": detection_data
                })

        return JSONResponse({
            "success": True,
            "camera_id": camera_id,
            "location": location,
            "model": "detection",
            "count": len(detections),
            "detections": detections
        })

    # ==========================================
    # Accident Model
    # ==========================================
    elif model_name == "accident":

        model = models["accident"]
        prediction = model.predict(processed_image, verbose=0)

        if prediction.shape[-1] == 1:
            pred = float(prediction[0][0])

            if pred > 0.5:
                class_name = "No Accident"
                confidence = pred
            else:
                class_name = "Accident"
                confidence = 1 - pred

        else:
            class_id = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            class_names = ["Accident", "No Accident"]
            class_name = class_names[class_id]

        result_data = {
            "camera_id": camera_id,
            "location": location,
            "model": "accident",
            "class": class_name,
            "confidence": round(confidence, 4)
        }

        if confidence > 0.75:
            await manager.broadcast({
                "type": "ALERT",
                "camera_id": camera_id,
                "location": location,
                "model": "accident",
                "data": result_data
            })

        return JSONResponse({
            "success": True,
            "camera_id": camera_id,
            "location": location,
            "model": "accident",
            "prediction": result_data
        })

    # ==========================================
    # Congestion Model
    # ==========================================
    elif model_name == "congestion":

        model = models["congestion"]
        prediction = model.predict(processed_image, verbose=0)

        pred = float(prediction[0][0])

        if pred > 0.5:
            class_name = "NO CONGESTION"
            confidence = pred
        else:
            class_name = "CONGESTION"
            confidence = 1 - pred

        result_data = {
            "camera_id": camera_id,
            "location": location,
            "model": "congestion",
            "class": class_name,
            "confidence": round(confidence, 4)
        }

        if confidence > 0.75:
            await manager.broadcast({
                "type": "ALERT",
                "camera_id": camera_id,
                "location": location,
                "model": "congestion",
                "data": result_data
            })

        return JSONResponse({
            "success": True,
            "camera_id": camera_id,
            "location": location,
            "model": "congestion",
            "prediction": result_data
        })
