import cv2
import numpy as np
import time
from picamera2 import Picamera2
import torch
from torchvision import transforms
import cv2

# Load your model (adjust the path if needed)
model = torch.load("your_model.pt")
model.eval()

# Preprocessing transformations (adjust as needed for your model)
preprocess = transforms.Compose([
    transforms.ToPILImage(),                 # Convert OpenCV image (NumPy array) to PIL image
    transforms.Resize((224, 224)),           # Resize to model's input size
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

def preprocess_image(image):
    """
    Preprocesses the input image for model prediction.
    Args:
        image (numpy array): The captured image in BGR format (OpenCV format).
    Returns:
        torch.Tensor: The preprocessed image ready for model input.
    """
    # Convert BGR (OpenCV) to RGB format for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply transformations
    input_tensor = preprocess(image_rgb)
    # Add a batch dimension
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor

# Debug function to display captured image
def debug_show_image(image, title="Captured Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Parameters for detection area
x, y, w, h = 100, 100, 300, 300
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
        input_tensor = preprocess_image(roi)  # roi is the cropped image
        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_class = prediction.argmax().item()
        
        # Display captured ROI for debugging
        debug_show_image(roi)

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
