# 🚦 Traffic Sign Recognition from Live Video

##  About the Project
This project focuses on building a robust and high-performance **traffic sign recognition system**.  
The core of the system is a **Convolutional Neural Network (CNN)** trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.  
The project leverages **transfer learning** with a pre-trained **VGG16 model** and employs advanced techniques like **fine-tuning** and **class weighting** to achieve high accuracy, even on an imbalanced dataset.  

A key feature of this project is its ability to perform **real-time inference on a live video stream**. By integrating the trained model with an **object detection framework (YOLOv8)**, it can detect and classify traffic signs as they appear in the video feed from a webcam.  

---

##  Key Features
-  **High Accuracy**: Achieves ~97.77% validation accuracy.  
-  **Transfer Learning with VGG16**: Speeds up training and boosts performance.  
-  **Fine-Tuning**: Top layers fine-tuned on GTSRB dataset for better feature learning.  
-  **Class Weighting**: Handles dataset imbalance effectively.  
-  **Real-Time Detection**: Detects and classifies traffic signs from a webcam feed.  

---

##  Technologies Used
- **Python** – Main programming language  
- **TensorFlow & Keras** – Model building, training, evaluation  
- **OpenCV (cv2)** – Image & video processing (capture, resize, bounding boxes)  
- **Pandas** – Handling dataset annotations  
- **Scikit-learn** – Data splitting & utilities  
- **YOLOv8** – Object detection for locating signs before classification  

---

##  Dataset
The model is trained on the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset:  
- 50,000+ images  
- 43 traffic sign classes  
- Organized into **Train** (class subfolders) and **Test** directories with CSV annotations  

---

##  Training Results
The model was trained using a **two-stage approach**:

1. **Feature Extraction**  
   - Frozen VGG16 top layers  
   - Trained only new classification layers for **10 epochs**  

2. **Fine-Tuning**  
   - Unfrozen top 4 VGG16 layers  
   - Trained full model with low learning rate for **50 epochs**  

**Final Training Metrics (Epoch 50):**
- `accuracy: 0.9482`  
- `loss: 0.2507`  
- `val_accuracy: 0.9777`  
- `val_loss: 0.0664`  

⏹ Training stopped early as **validation loss kept improving**, reaching **0.0664** with **97.77% validation accuracy**.  

---

##  Real-Time Application
The final system integrates **YOLOv8 detection + CNN classification**, enabling **real-time traffic sign recognition** from live webcam feeds.  
The trained model is loaded, you can use it by setting the file paths as specified in the code.
There is also a code available that allows you to classify photos by uploading them directly.

---



