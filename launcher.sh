#!/bin/sh
# launcher.sh

# Step 1: Activate the virtual environment
source /path/to/tflite-env/bin/activate

# Step 2: Navigate to the object_detection folder
cd /path/to/object_detection

# Step 3: Run the main_object_detection script
python main_object_detection.py

# Keep the terminal open if an error occurs
echo "Program terminated. Press Enter to close."
read
