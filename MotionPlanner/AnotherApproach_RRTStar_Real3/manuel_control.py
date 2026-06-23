import serial
import struct
import sys

# --- CONFIGURATION ---
UART_PORT = '/dev/ttyAMA0'
BAUD_RATE = 115200
SYNC_BYTE_1 = 0xAA
SYNC_BYTE_2 = 0xBB

def calculate_checksum(payload_bytes):
    return sum(payload_bytes) & 0xFF

def main():
    # 1. Open UART Connection
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Connected to STM32 on {UART_PORT} at {BAUD_RATE} baud.")
        ser.reset_input_buffer()
    except Exception as e:
        print(f"[ERROR] Could not open UART: {e}")
        sys.exit(1)

    print("\n" + "="*40)
    print("      SNAKE ROBOT MANUAL CONTROL")
    print("="*40)
    print("Enter commands as 5 numbers separated by spaces:")
    print("Format:  [Dist_Head] [Dist_Link2] [Q1_Angle] [Q2_Angle] [Q3_Angle]")
    print("Example: 2.5 2.5 15.0 -10.0 0.0")
    print("Type 'q', 'quit', or 'exit' to stop.")
    print("="*40 + "\n")

    # Clear out any leftover REQ messages from previous runs
    ser.reset_input_buffer()

    while True:
        try:
            # 2. Get User Input
            user_input = input(">> Command: ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            parts = user_input.split()
            if len(parts) != 5:
                print("[ERROR] Please provide exactly 5 numbers.")
                continue
            
            dist_head = float(parts[0])
            dist_link2 = float(parts[1])
            q1 = float(parts[2])
            q2 = float(parts[3])
            q3 = float(parts[4])
            
            # 3. Pack the Payload (Mirroring the STM32 Struct)
            duration_ms = 1000 # STM32 ignores this and uses physical completion anyway
            q1_pitch, q2_pitch, q3_pitch = 0.0, 0.0, 0.0
            
            payload = struct.pack('<H 8f', 
                                  duration_ms, 
                                  dist_head, dist_link2, 
                                  q1, q2, q3, 
                                  q1_pitch, q2_pitch, q3_pitch)
            
            checksum = calculate_checksum(payload)
            packet = bytes([SYNC_BYTE_1, SYNC_BYTE_2]) + payload + bytes([checksum])
            
            # 4. Send to STM32
            ser.write(packet)
            ser.flush()
            print(f"[TX] Sent -> DC Units: ({dist_head}, {dist_link2}) | Servos: ({q1}°, {q2}°, {q3}°)")
            
            # 5. Wait for STM32 to finish physical movement
            print("[WAIT] Robot is moving... Waiting for confirmation...")
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if "REQ" in line:
                    print(f"[RX] Move Complete! Odometry: {line}\n")
                    break
                elif line:
                    print(f"[STM32 DEBUG] {line}")
                    
        except ValueError:
            print("[ERROR] Invalid input. Please enter numbers only (e.g., '1.5 -2.0 10 0 0').\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] Communication error: {e}\n")
            
    # 6. Send End Sentinel before closing
    print("\n[INFO] Sending END dummy packet to STM32...")
    dummy_payload = struct.pack('<H 8f', 65535, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dummy_checksum = calculate_checksum(dummy_payload)
    dummy_packet = bytes([SYNC_BYTE_1, SYNC_BYTE_2]) + dummy_payload + bytes([dummy_checksum])
    try:
        ser.write(dummy_packet)
        ser.close()
    except:
        pass
    print("[INFO] Exited Manual Mode.")

if __name__ == "__main__":
    main()