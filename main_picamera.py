import cv2
import numpy as np
import time
from picamera2 import Picamera2
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from PIL import Image

# --- Model ---
# Load the TFLite model and allocate tensors
interpreter = Interpreter(model_path="best_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Image preprocessing function
def preprocess_image(image):
    input_shape = input_details[0]['shape']
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32) / 255.0
    return image

# Function for making predictions
def predict_tflite(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    return "Class 1" if prediction == 1 else "Class 0"

# --- Camera ---
# Debug function to display captured image
def debug_show_image(image,pred, title="Captured Image"):
    # Display prediction on frame
    cv2.putText(frame, pred, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 1000)})
picam2.configure(config)
picam2.start()

# Parameters for detection area
x, y, w, h = 150, 560, 300, 300
delay_seconds = 10  # Set delay before capture
capture_start_time = None  # Timer variable

while True:
    # Capture frame from PiCamera2
    frame = picam2.capture_array()

    # Draw outer green rectangle for subject detection area
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display current time
    cv2.putText(frame, f"{time.ctime()}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Check if delay time has passed
    if capture_start_time is None:
        capture_start_time = time.time()  # Start the timer

    # Capture after delay
    if time.time() - capture_start_time >= delay_seconds:
        # Crop and preprocess the region of interest (ROI) inside the boundary
        roi = frame[y:y+h, x:x+w]
        # Process or classify the ROI here (e.g., pass it to a model)
        subject_img = preprocess_image(roi)
        prediction = predict_tflite(subject_img)

        # Display prediction on frame
        cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display captured ROI for debugging
        debug_show_image(roi,prediction)

        # Reset timer to wait before the next capture
        capture_start_time = time.time()

    # Display frame with bounding box
    cv2.imshow('Camera', frame)

    # Break the loop with 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
