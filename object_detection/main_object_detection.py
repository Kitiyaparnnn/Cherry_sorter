import threading
import queue
import time
from time import sleep
import cv2
import numpy as np
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter
import RPi.GPIO as GPIO

# Stop signal (shared event)
stop_event = threading.Event()

# GPIO port setup
GPIO.setmode(GPIO.BOARD)
ir_sensor_gpio = 5
servo_moter = 11

# --- IR Sensor Setup ---
GPIO.setup(ir_sensor_gpio,GPIO.IN)

# --- Servo Setup ---
GPIO.setup(servo_moter, GPIO.OUT)
servo = GPIO.PWM(servo_moter, 50)
servo.start(2.5)


def servo_movement(prediction_queue, stop_event):
    """
    Control the servo based on predictions from the queue.
    Result: 0-bad, 1-good
    """
    ir = GPIO.input(ir_sensor_gpio)
    ir_count = 0
    while not stop_event.is_set():
        try:
            
            if ir != GPIO.LOW:
                ir_count += 1
            if ir_count > 4:
                # Get the next prediction from the queue
                prediction = prediction_queue.get_nowait()  # Non-blocking, raises queue.Empty if empty
                
                # Move the servo based on the prediction when ir sensor triggers
                if np.any(prediction == 0):
                    servo.ChangeDutyCycle(12.5)  # Move right
                    sleep(0.5)
                    servo.ChangeDutyCycle(7.5)  # Center position
                # servo.ChangeDutyCycle(0)  # Stop servo
                
                ir_count = 0
            sleep(0.5)
            
        except queue.Empty:
            continue

# --- Prediction Setup ---
PATH_TO_LABELS = 'custom_model_lite_new2/labelmap.txt'
PATH_TO_MODEL = 'custom_model_lite_new2/detect.tflite'

# Load the label map into memory
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the Tensorflow Lite model into memory
interpreter = Interpreter(model_path=PATH_TO_MODEL)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


float_input = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

     # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    return input_data

# --- Camera Setup ---
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (500,500), "format": "RGB888"})
picam2.configure(config)
picam2.start()

#x, y, w, h = 140, 60, 200, 200
x, y, w, h = 0, 0, 500,500
delay = 0.5 #scond unit
min_conf = 0.7

# --- Image Classification Thread ---
def image_classification(prediction_queue, stop_event):
    """Capture images, make predictions, and add them to the queue."""

    good_cherry = 0
    bad_cherry = 0
    
    while not stop_event.is_set():
        # Capture frame from PiCamera2
        frame = picam2.capture_array()
        roi = frame[y:y+h, x:x+w]
        # Draw outer green rectangle for subject detection area
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
        
        #--chage frame
        imH, imW, _ = roi.shape
        
        # Perform the actual detection by running the model with the image as input
        #--chage frame
        input_data = preprocess_image(roi) 
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
        
        print("Score:", scores)


        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))

                #--chage frame
                cv2.rectangle(roi, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2) 

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                #--chage frame
                cv2.rectangle(roi, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                #--chage frame
                cv2.putText(roi, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    
                # Add the prediction to the queue
                if not prediction_queue.full():  # Avoid blocking if queue is full
                    prediction_queue.put(classes[i])
                        
                else:
                    print("Prediction queue is full. Dropping oldest prediction.")
                    prediction_queue.get()  # Remove the oldest prediction
                    prediction_queue.put(classes[i])
                    
                # Count cherry
                if classes[i] == 1:
                    good_cherry += 1
                else:
                    bad_cherry += 1
        
        # Display frame with bounding box
        cv2.imshow('Camera', frame)
        
        print("qeue: ", prediction_queue.queue)
        sleep(delay) #second unit
        
        # Check for 'q' key press to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping threads...")
            stop_event.set()
            break
        
    print(f"---\n Summary:\n good cherries: {good_cherry}, bad cherries: {bad_cherry}\n ---\n")

# --- Main Program ---
if __name__ == "__main__":
    prediction_queue = queue.Queue()  # Queue to share predictions between threads

    # Create and start threads
    prediction_thread = threading.Thread(target=image_classification, args=(prediction_queue,stop_event))
    servo_thread = threading.Thread(target=servo_movement, args=(prediction_queue, stop_event))
    prediction_thread.start()
    servo_thread.start()

    # Join thrveads (optional, for cleanup on program termination)
    prediction_thread.join()
    servo_thread.join()


    # Cleanup
    picam2.stop()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()
