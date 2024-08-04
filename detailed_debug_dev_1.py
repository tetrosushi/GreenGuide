import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define paths
train_dir = r"C:\Users\susha\Downloads\dataset\dataset\train"
validation_dir = r"C:\Users\susha\Downloads\dataset\dataset\validation"
model_checkpoint_path = r"C:\Users\susha\Downloads\recycling_model.keras"
weights_save_path = r"C:\Users\susha\Downloads\recycling_model.weights.h5"

# Load pre-trained MobileNetV2 and fine-tune for recycling objects classification
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Assuming 3 classes: compost, garbage, recycling
model = Model(inputs=base_model.input, outputs=predictions)

# Data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set up checkpoints and early stopping
checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping]
)

# Save the trained model weights
model.save_weights(weights_save_path)

# Set up the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1240)
cap.set(4, 720)

# Preprocess input for the model
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_array = img_to_array(frame_resized)
    frame_preprocessed = preprocess_input(frame_array)
    return np.expand_dims(frame_preprocessed, axis=0)

# Load class labels
class_labels = ['Compost', 'Garbage', 'Recycling']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    frame = cv2.flip(frame, 1)

    # Preprocess the frame for the model
    preprocessed_frame = preprocess_frame(frame)

    # Predict the class
    predictions = model.predict(preprocessed_frame)
    class_id = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Display the result
    label = f'{class_labels[class_id]}: {confidence:.2f}'
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Recycling Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
