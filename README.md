# Cherry_sorting

**Dataset**

- good: a good cherry image
- bad: a bad cherry image
- multiple: a multiple cherries image

**Library**

Make sure TensorFlow Lite runtime is installed on the Raspberry Pi. You can install it with:

`pip3 install tflite-runtime`

### Step 1: Activate a virtual environment on Raspberry Pi named "tflite-env"

`source tflite-env/bin/activate`

### Step 2: Load a custom model tflite folder

`git clone https://github.com/Kitiyaparnnn/Cherry_sorter.git`

`cd Cherry_sorter`

`cp -r object_detection/custom_model_lite ../custom_model/lite`

### Step 3: redirect to the project file

`cd custom_model_lite/`

### Step 3: Run main_picamera.py

`python main_object_detection.py`