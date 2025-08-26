import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from ultralytics import YOLO

# --- 1. Load the Models ---

# Load the custom trained traffic sign classification model
model_path = 'final_traffic_sign_model.h5'
try:
    classifier_model = keras.models.load_model(model_path)
    print("Classification model successfully loaded.")
except Exception as e:
    print(f"Error: Classification model could not be loaded. Make sure 'best_traffic_sign_model.h5' is in the correct path.\nError: {e}")
    exit()

# Load the pre-trained YOLOv8 model
yolo_model = YOLO('yolov8n.pt')
print("YOLOv8 model successfully loaded.")

# --- 2. Define the Classes ---

# GTSRB's 43 classes
gtsrb_class_names = {
    0: "Speed Limit (20km/h)", 1: "Speed Limit (30km/h)", 2: "Speed Limit (50km/h)",
    3: "Speed Limit (60km/h)", 4: "Speed Limit (70km/h)", 5: "Speed Limit (80km/h)",
    6: "End of Speed Limit (80km/h)", 7: "Speed Limit (100km/h)", 8: "Speed Limit (120km/h)",
    9: "No passing", 10: "No passing for vehicles over 3.5 metric tons", 11: "Right-of-way at intersection",
    12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited", 17: "No entry", 18: "General caution",
    19: "Dangerous curve to the left", 20: "Dangerous curve to the right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on the right",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians",
    28: "Children crossing", 29: "Bicycles crossing", 30: "Beware of ice/snow",
    31: "Wild animals crossing", 32: "End of all speed and passing limits",
    33: "Turn right ahead", 34: "Turn left ahead", 35: "Ahead only",
    36: "Go straight or right", 37: "Go straight or left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End of no passing", 42: "End of no passing by vehicles over 3.5 metric tons"
}

def predict_from_frame(frame, x1, y1, x2, y2):
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "Not Found", 0.0

    image_resized = cv2.resize(roi, (48, 48))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    image_preprocessed = preprocess_input(image_rgb)
    
    image_final = np.expand_dims(image_preprocessed, axis=0)
    
    predictions = classifier_model.predict(image_final, verbose=0)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    
    predicted_label = gtsrb_class_names.get(predicted_class_index, "Unknown")
    
    return predicted_label, confidence

# --- 3. Start Live Webcam Feed ---

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    yolo_results = yolo_model(frame, verbose=False)
    
    for r in yolo_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = box.cls
            
            if box.conf > 0.5:
                predicted_label, confidence = predict_from_frame(frame, x1, y1, x2, y2)
                
                display_text = f"{predicted_label} ({confidence:.2f})"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Live Traffic Sign Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()