import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys


print("Loading the model required for classification...")
model_path = 'final_traffic_sign_modelYeni.h5'

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    classifier_model = keras.models.load_model(model_path)
    print("The model was loaded successfully.")
except Exception as e:
    messagebox.showerror("Model Error", f"Model yüklenemedi. Lütfen 'best_traffic_sign_model.h5' dosyasının doğru yerde olduğundan emin olun.\nHata: {e}")
    sys.exit()

image_size = (48, 48)

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

def predict_image(image_path):
    """
    Verilen fotoğraf yolunu alır, model için hazırlar ve sınıflandırma yapar.
    """
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError

        image_resized = cv2.resize(frame, image_size)
        
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        image_preprocessed = preprocess_input(image_rgb)

        image_final = np.expand_dims(image_preprocessed, axis=0)

        print("\nThe photo is classified...")
        predictions = classifier_model.predict(image_final, verbose=0)
        
        predicted_class_index = np.argmax(predictions)
        confidence = np.max(predictions)

        predicted_label = gtsrb_class_names.get(predicted_class_index, "Unknown")

        print("--- Classification Result ---")
        print(f"Detected Plate: {predicted_label}")
        print(f"Confidence Score: {confidence:.2f}")

        display_text = f"Plate: {predicted_label}"
        confidence_text = f"Confidence: {confidence:.2f}"
        
        orig_height, orig_width = frame.shape[:2]
        ratio = 600 / max(orig_height, orig_width)
        resized_frame = cv2.resize(frame, (int(orig_width * ratio), int(orig_height * ratio)))
        
        cv2.putText(resized_frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(resized_frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Traffic Sign Recognition', resized_frame)
        print("\nThe process is complete. Press any key to close the display window.")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError:
        messagebox.showerror("File Eror", "The selected file was not found or is invalid.")
    except Exception as e:
        messagebox.showerror("Process Error", f"An error occurred during processing: {e}")
        
def select_image_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a traffic sign photo to classify",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.ppm")]
    )
    return file_path

if __name__ == "__main__":
    selected_path = select_image_path()
    if selected_path:
        predict_image(selected_path)
    else:
        print("No photo selected. Program is terminating.")