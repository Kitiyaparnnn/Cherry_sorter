#!/bin/sh
# launcher.sh
# navigate to home directory, then to this directory, then execute python script, then back home

cd /
cd home/coffeecolor
source tflite-env/bin/activate
python main_object_detection.py
cd /