import serial

# On Pi 5, the physical pins are often ttyAMA0, not serial0
try:
    ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
    print("Listening to physical GPIO pins... Press Ctrl+C to stop.")
    while True:
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting)
            print(f"I HEARD SOMETHING: {data}")
except Exception as e:
    print(f"Error: {e}")