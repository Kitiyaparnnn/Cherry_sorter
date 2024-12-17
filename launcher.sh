#!/bin/sh
# launcher.sh

# Step 1: Activate the virtual environment
source /home/coffeecolor/tflite-env/bin/activate

# Step 2: Navigate to the object_detection folder
cd /home/coffeecolor/Cherry_sorter/object_detection

# Step 3: Run the main_object_detection script
python main.py

# Keep the terminal open if an error occurs
echo "Program terminated. Press Enter to close."
read

# Setup launcher @systemd 
# Make the script executable:
# `chmod +x launcher.sh`
# Create a service file for your script:
# `sudo nano /etc/systemd/system/object_detection.service`
# Add the following content:
# `
# [Unit]
# Description=Object Detection Launcher
# After=graphical.target

# [Service]
# Type=simple
# ExecStart=/path/to/launcher.sh
# Restart=on-failure
# User=coffeecolor
# Environment=DISPLAY=:0
# Environment=XAUTHORITY=/home/coffeecolor/.Xauthority

# StandardOutput=append:/home/coffeecolor/launcher.log
# StandardError=append:/home/coffeecolor/launcher.log

# [Install]
# WantedBy=graphical.target

# `
# Other options for Restart include:
# on-failure ? Restarts only if the program crashes (non-zero exit code).
# always ? Restarts no matter why the program exited (default behavior).
# on-success ? Restarts only if the program exits cleanly.
# Enable and start the service:
# `
# sudo systemctl enable object_detection.service
# sudo systemctl start object_detection.service

# `
# Verify the service is running:
# `sudo systemctl status object_detection.service`
# Reboot to ensure it starts automatically:
# `sudo reboot`