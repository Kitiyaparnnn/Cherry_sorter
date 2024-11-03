import cv2
import numpy as np
import time
import torch  # Use PyTorch or change to `tensorflow.keras.models` if using TensorFlow
# from torchvision import transforms

# Load your model (adjust path and model structure if needed)
# model = torch.load("your_model.pt")  # For PyTorch, adjust if TensorFlow
# model.eval()

# # Preprocessing transformations (adjust as needed for your model)
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize to model's input size
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Mean and STD for pretrained models
# ])

# Debug function to display captured image
def debug_show_image(image, title="Captured Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

# Start video capture from camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera
cap.set(cv2.CAP_PROP_FPS, 10)

# Parameters for detection area
x, y, w, h = 100, 100, 300, 300
delay_seconds = 10  # Set delay before capture
capture_start_time = None  # Timer variable

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw outer green rectangle for subject detection area
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Crop the region of interest (ROI) inside the boundary
    roi = frame[y:y+h, x:x+w]

    cv2.putText(frame, f"{time.ctime()}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Check if delay time has passed
    if capture_start_time is None:
        capture_start_time = time.time()  # Start the timer

    # Capture after delay
    if time.time() - capture_start_time >= delay_seconds:
        # Crop and preprocess the region of interest (ROI) inside the boundary
        roi = frame[y:y+h, x:x+w]
        subject_img = roi
        # prediction = predict_tflite(subject_img)

        # Display prediction on frame
        # cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display captured image
        debug_show_image(subject_img)

        # Reset timer to wait before the next capture
        capture_start_time = time.time()

    # Display frame with bounding box
    cv2.imshow('Camera', frame)

    # Break the loop with 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
