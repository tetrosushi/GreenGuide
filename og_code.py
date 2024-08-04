import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array

# Load pre-trained MobileNetV2 and fine-tune for recycling objects classification
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)  # Assuming 10 classes of recycling objects
model = Model(inputs=base_model.input, outputs=predictions)

# Load the fine-tuned model weights if available
# model.load_weights('recycling_model.h5') # Uncomment if model weights are saved

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
class_labels = ['Plastic', 'Glass', 'Metal', 'Paper', 'Cardboard', 'Trash', 'Battery', 'Electronic', 'Compost', 'Other']

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