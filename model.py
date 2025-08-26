import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter



# Read training CSV file an set file path (use your file path here)
train_csv_path = '/content/GTSRB_Final_Training_annotations.csv'
if not os.path.exists(train_csv_path):
    print("Hata: 'GTSRB_Final_Training_annotations.csv' dosyası bulunamadı. Lütfen yüklediğinizden emin olun.")
    exit()
train_df = pd.read_csv(train_csv_path)


gtsrb_train_path = '/content/gtsrb_train'
train_df['Path'] = train_df['Path'].apply(lambda x: os.path.join(gtsrb_train_path, x))

# We are making data set arrangements
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['ClassId'])

print(f"Total number of files: {len(train_df) + len(val_df)}")
print(f"Number of files allocated for training: {len(train_df)}")
print(f"Number of files allocated for verification: {len(val_df)}")

# --- 2. Calculate and Use Class Weights ---

class_counts = Counter(train_df['ClassId'])
total_samples = len(train_df)
num_classes = len(class_counts)

max_samples = max(class_counts.values())
class_weights = {class_id: max_samples / count for class_id, count in class_counts.items()}

print("\nHesaplanan Sınıf Ağırlıkları (Class Weights):")
for class_id, weight in sorted(class_weights.items()):
    print(f"Sınıf {class_id}: {weight:.2f}")

image_size = (48, 48)
batch_size = 32

# Data augmention settings
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=False,
    brightness_range=[0.5, 1.5],  
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Path',
    y_col='ClassId',
    target_size=image_size,
    color_mode='rgb',
    class_mode='raw',
    batch_size=batch_size,
    shuffle=True
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='Path',
    y_col='ClassId',
    target_size=image_size,
    color_mode='rgb',
    class_mode='raw',
    batch_size=batch_size,
    shuffle=False
)

# --- 4. Model Creation and Training ---

# VGG16 
base_model = VGG16(
    input_shape=(image_size[0], image_size[1], 3),
    include_top=False,
    weights='imagenet'
)


base_model.trainable = False
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nAşama 1: Yeni katmanlar eğitiliyor...")
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    class_weight=class_weights,  # <-- BURADA KULLANIYORUZ
    verbose=1
)

# Second stage: Fine-tuning
base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath='best_traffic_sign_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

print("\nAşama 2: Modelin tamamına fine-tuning yapılıyor...")
model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint],
    class_weight=class_weights,  
    verbose=1
)

model.save('final_traffic_sign_model.h5')
print("The model was trained successfully and saved as 'final_traffic_sign_model.h5'.")