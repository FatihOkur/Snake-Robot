import serial
import json
import time
import struct
import sys

# --- CONFIGURATION ---
JSON_FILE = 'robot_path_commands.json'
UART_PORT = '/dev/ttyAMA10'  # Default Pi 5 UART pin (GPIO 14/15). Change to /dev/ttyUSB0 if using a USB-TTL adapter.
BAUD_RATE = 115200

# STM32 signals
STM32_REQ_SIGNAL = b'REQ'   # STM32 sends this when it has space in its ring buffer
SYNC_BYTE_1 = 0xAA          # Header to prevent STM32 from reading misaligned data
SYNC_BYTE_2 = 0xBB

def calculate_checksum(payload_bytes):
    """Calculates a simple 8-bit checksum for data integrity."""
    return sum(payload_bytes) & 0xFF

def main():
    # 1. Load the Trajectory
    try:
        with open(JSON_FILE, 'r') as f:
            raw_trajectory = json.load(f)
        print(f"[INFO] Loaded {len(raw_trajectory)} raw steps from {JSON_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not load JSON: {e}")
        sys.exit(1)

    # 1.5 Stutter Filter: Remove mid-path steps where distance is 0.0 AND no angles change
    trajectory = []
    for i, cmd in enumerate(raw_trajectory):
        dist_head = float(cmd["dc_motor_commands"]["segment1_head_distance_units"])
        dist_link2 = float(cmd["dc_motor_commands"]["segment3_link2_distance_units"])
        
        q1_yaw = float(cmd["servo_yaw_commands"]["q1_deg"])
        q2_yaw = float(cmd["servo_yaw_commands"]["q2_deg"])
        q3_yaw = float(cmd["servo_yaw_commands"]["q3_deg"])

        angle_changed = False
        if trajectory:
            last_cmd = trajectory[-1]
            last_q1 = float(last_cmd["servo_yaw_commands"]["q1_deg"])
            last_q2 = float(last_cmd["servo_yaw_commands"]["q2_deg"])
            last_q3 = float(last_cmd["servo_yaw_commands"]["q3_deg"])
            if abs(q1_yaw - last_q1) > 0.1 or abs(q2_yaw - last_q2) > 0.1 or abs(q3_yaw - last_q3) > 0.1:
                angle_changed = True
        else:
            angle_changed = True # Always keep the very first step to initialize pose
        
        # Keep the step if it requires movement, OR if angle changed, OR if it's the very last step
        if dist_head != 0.0 or dist_link2 != 0.0 or angle_changed or i == (len(raw_trajectory) - 1):
            trajectory.append(cmd)
            
    total_steps = len(trajectory)
    print(f"[INFO] Filtered trajectory down to {total_steps} active steps.")

    # 2. Open UART Connection
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Connected to STM32 on {UART_PORT} at {BAUD_RATE} baud.")
        time.sleep(2) # Brief pause to allow UART to stabilize
    except Exception as e:
        print(f"[ERROR] Could not open UART: {e}")
        sys.exit(1)

    step_index = 0

    print("[INFO] Waiting for STM32 to request the first step...")

    # 3. Main Communication Loop
    try:
        while step_index < total_steps:
            # Listen for the STM32's request signal ("REQ")
            if ser.in_waiting > 0:
                incoming = ser.readline().strip()
                
                if incoming == STM32_REQ_SIGNAL:
                    print(f"[TX] STM32 requested data. Sending Step {step_index + 1}/{total_steps}...")
                    
                    cmd = trajectory[step_index]
                    
                    # Extract values based on new JSON schema
                    duration_ms = int(cmd["step_duration_ms"])
                    dist_head = float(cmd["dc_motor_commands"]["segment1_head_distance_units"])
                    dist_link2 = float(cmd["dc_motor_commands"]["segment3_link2_distance_units"])
                    
                    # Yaw (Lateral)
                    q1_yaw = float(cmd["servo_yaw_commands"]["q1_deg"])
                    q2_yaw = float(cmd["servo_yaw_commands"]["q2_deg"])
                    q3_yaw = float(cmd["servo_yaw_commands"]["q3_deg"])
                    
                    # Pitch (Climbing)
                    q1_pitch = float(cmd["servo_pitch_commands"]["q1_pitch_deg"])
                    q2_pitch = float(cmd["servo_pitch_commands"]["q2_pitch_deg"])
                    q3_pitch = float(cmd["servo_pitch_commands"]["q3_pitch_deg"])

                    # 4. Pack into a compact binary struct
                    # '<' = Little Endian (Standard for STM32 ARM Cortex)
                    # 'H' = unsigned short (2 bytes) for duration
                    # '8f' = 8 floats (32 bytes) for distances and angles
                    payload = struct.pack('<H 8f', 
                                          duration_ms, 
                                          dist_head, dist_link2, 
                                          q1_yaw, q2_yaw, q3_yaw, 
                                          q1_pitch, q2_pitch, q3_pitch)

                    checksum = calculate_checksum(payload)
                    
                    # Construct final packet: [SYNC1, SYNC2, PAYLOAD, CHECKSUM]
                    packet = bytes([SYNC_BYTE_1, SYNC_BYTE_2]) + payload + bytes([checksum])
                    
                    # 5. Transmit
                    ser.write(packet)
                    ser.flush()
                    
                    step_index += 1
                else:
                    # Optional: Print debug messages sent from STM32
                    print(f"[STM32 DEBUG] {incoming.decode('utf-8', errors='ignore')}")

        # Trajectory Complete
        print("[INFO] Trajectory complete. Sending END dummy packet.")
        # Send dummy packet with duration = 0xFFFF (65535) and all zeroes
        dummy_payload = struct.pack('<H 8f', 65535, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        dummy_checksum = calculate_checksum(dummy_payload)
        dummy_packet = bytes([SYNC_BYTE_1, SYNC_BYTE_2]) + dummy_payload + bytes([dummy_checksum])
        
        ser.write(dummy_packet)
        ser.flush()
        ser.close()

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Closing port.")
        ser.close()

if __name__ == "__main__":
    main()