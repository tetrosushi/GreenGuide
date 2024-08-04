# GreenGuide
# Recycling Object Detection

This project uses a fine-tuned MobileNetV2 model to classify recycling objects in real-time using a webcam.

## Prerequisites

Make sure you have the following libraries installed:

- OpenCV
- NumPy
- TensorFlow
- Keras

You can install them using pip:

```bash
pip install opencv-python-headless numpy tensorflow keras
```

## Description

The script captures video from the webcam, processes each frame, and uses a pre-trained MobileNetV2 model to classify the objects into one of 10 recycling categories:

- Plastic
- Glass
- Metal
- Paper
- Cardboard
- Trash
- Battery
- Electronic
- Compost
- Other

## Usage

1. **Load Pre-trained Model:**
   The script uses MobileNetV2 pre-trained on ImageNet and fine-tunes it for classifying recycling objects. The model expects weights to be loaded if available.

   Uncomment the following line if you have pre-trained weights saved as `recycling_model.h5`:
   ```python
   model.load_weights('recycling_model.h5')
   ```

2. **Run the Script:**
   To start the recycling object detection, simply run the script:
   ```bash
   python og_code.py
   ```

3. **Webcam Feed:**
   The script will open a webcam feed window where it will display the detected object and its confidence score. To quit, press the `q` key.

## Functions

- **`preprocess_frame(frame)`**: Resizes the frame to 224x224, converts it to an array, preprocesses it using the `preprocess_input` function from MobileNetV2, and expands the dimensions to match the input shape of the model.

## Classes

The model classifies objects into the following classes:
- Plastic
- Glass
- Metal
- Paper
- Cardboard
- Trash
- Battery
- Electronic
- Compost
- Other

## Additional Notes

- Ensure your webcam is properly connected.
- The frame size for the webcam feed is set to 1240x720. Adjust these settings if necessary.

## Acknowledgments

- This project uses the MobileNetV2 architecture from TensorFlow/Keras.
- OpenCV is used for capturing and displaying the webcam feed.

Feel free to contribute to this project by improving the model, adding new features, or fixing any bugs.

## License

This project is licensed under the MIT License.
