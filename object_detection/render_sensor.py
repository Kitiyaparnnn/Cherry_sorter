import serial

def render_sensor():
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=0.5)
    ser.reset_input_buffer()

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            return line


