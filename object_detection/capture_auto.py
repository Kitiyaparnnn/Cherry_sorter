import os
from picamera2 import Picamera2
import time
import cv2  # OpenCV for displaying images

# Define the folder to save images
SAVE_FOLDER = "captured_images"

def setup_folder():
    """Create the folder if it does not exist."""
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

def capture_and_display(camera, image_index):
    """Capture an image, save it to the folder, and display it."""
    # Generate a timestamped filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(SAVE_FOLDER, f"image_{image_index}_{timestamp}.jpg")
    
    # Capture and save the image
    camera.capture_file(filename)
    
    # Display the captured image for a short duration
    image = cv2.imread(filename)
    if image is not None:
        cv2.imshow("Captured Image", image)
        cv2.waitKey(1)  # Display the image briefly to update the window
    else:
        print("Error: Could not load the captured image for display.")
    
    return filename

def main():
    # Setup folder for saving images
    setup_folder()

    # Initialize Picamera2
    camera = Picamera2()
    camera.configure(camera.create_still_configuration(main={"size": (480, 1080)}))
    camera.start()
    
    print("Capturing images automatically at 30 FPS. Press 'q' to quit.")
    image_index = 1  # Counter for image numbering
    
    start_time = time.time()
    fps_interval = 1 / 30  # 30 FPS (time per frame)
    
    try:
        while True:
            current_time = time.time()
            if current_time - start_time >= fps_interval:
                # Capture and display the image
                filename = capture_and_display(camera, image_index)
                print(f"Image saved: {filename}")
                
                image_index += 1  # Increment image counter
                start_time = current_time
            
            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting program...")
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
