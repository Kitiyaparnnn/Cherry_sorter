# 🍒Coffee Cherry Color Sorting 

**Dataset**

- good: a good cherry image
- bad: a bad cherry image
- multiple: a multiple cherries image

**Library**

Make sure TensorFlow Lite runtime is installed on the Raspberry Pi. You can install it with:

tflite : `pip3 install tflite-runtime`

pillow : `pip3 install pillow`


### Step 1: Activate a virtual environment on Raspberry Pi named "tflite-env"

`source tflite-env/bin/activate`

### Step 2: Load and copy an object detection folder

`git clone https://github.com/Kitiyaparnnn/Cherry_sorter.git`

`cd Cherry_sorter`

`cp -r object_detection ../object_detection`

### Step 3: Redirect to the project file

`cd object_detection/`

### Step 3: Run program

- v.1 (using photoelectric sensor) : `python main.py`
- v.2 (using timer delay) : `python main_wihtout_sensor.py`

