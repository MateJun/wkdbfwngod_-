import serial
import time

# --- Python script's Arduino settings from run_ufld_drive.py ---
PORT = 'COM9'
BAUD = 9600
SERVO_CENTER = 460
SERVO_MIN = 384
SERVO_MAX = 550
STOP_SIGNAL = 0

print(f"Connecting to Arduino on {PORT} at {BAUD} baud...")
try:
    arduino = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # Give Arduino time to reset
    print("Arduino connected successfully.")
except Exception as e:
    print(f"Failed to connect to Arduino: {e}")
    print("Please check the COM port, ensure Arduino IDE Serial Monitor is closed, and Arduino is connected.")
    exit()

def send_steer_command(steer_val):
    command = f"{steer_val}\n"
    print(f"Sending command: '{command.strip()}'")
    arduino.write(command.encode())
    time.sleep(0.1) # Short delay for Arduino to process

try:
    print("\n--- Testing steering motor ---")
    
    # Test Center
    print("Setting steering to CENTER...")
    send_steer_command(SERVO_CENTER)
    time.sleep(2)

    # Test MIN (Left)
    print("Setting steering to MIN (Left)...")
    send_steer_command(SERVO_MIN)
    time.sleep(2)

    # Test MAX (Right)
    print("Setting steering to MAX (Right)...")
    send_steer_command(SERVO_MAX)
    time.sleep(2)

    # Return to Center
    print("Returning steering to CENTER...")
    send_steer_command(SERVO_CENTER)
    time.sleep(2)

    # Test Stop Signal
    print("Sending STOP signal (0)...")
    send_steer_command(STOP_SIGNAL)
    time.sleep(1)

    print("\n--- Test complete ---")

except KeyboardInterrupt:
    print("\nTest interrupted by user.")
finally:
    print("Closing Arduino connection and sending STOP signal...")
    send_steer_command(STOP_SIGNAL) # Ensure motors are stopped
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
    print("Connection closed.")
