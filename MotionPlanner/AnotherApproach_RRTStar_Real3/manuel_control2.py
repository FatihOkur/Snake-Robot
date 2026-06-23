import serial
import struct
import sys
import math

# --- CONFIGURATION ---
UART_PORT = '/dev/ttyAMA0'
BAUD_RATE = 115200
SYNC_BYTE_1 = 0xAA
SYNC_BYTE_2 = 0xBB

# The maximum allowed distance per synchronized chunk (Planner Units)
MAX_STEP_UNITS = 0.5 

def calculate_checksum(payload_bytes):
    return sum(payload_bytes) & 0xFF

def main():
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Connected to STM32 on {UART_PORT} at {BAUD_RATE} baud.")
        ser.reset_input_buffer()
    except Exception as e:
        print(f"[ERROR] Could not open UART: {e}")
        sys.exit(1)

    print("\n" + "="*45)
    print("      SNAKE ROBOT MANUAL CONTROL (SYNCED)")
    print("="*45)
    print("Enter commands as 5 numbers separated by spaces:")
    print("Format:  [Dist_Head] [Dist_Link2] [Q1] [Q2] [Q3]")
    print("Example: 5.0 5.0 0.0 0.0 0.0")
    print("Type 'q' to stop.")
    print("="*45 + "\n")

    ser.reset_input_buffer()

    while True:
        try:
            user_input = input("\n>> Command: ").strip()
            if user_input.lower() in ['q', 'quit', 'exit']: break
            
            parts = user_input.split()
            if len(parts) != 5:
                print("[ERROR] Please provide exactly 5 numbers.")
                continue
            
            total_dist_head = float(parts[0])
            total_dist_link2 = float(parts[1])
            total_q1 = float(parts[2])
            total_q2 = float(parts[3])
            total_q3 = float(parts[4])
            
            # --- 1. CALCULATE SLICES FOR SYNCHRONIZATION ---
            # Find the longest distance requested to determine how many chunks we need
            max_dist = max(abs(total_dist_head), abs(total_dist_link2))
            num_slices = max(1, int(math.ceil(max_dist / MAX_STEP_UNITS)))
            
            if num_slices > 1:
                print(f"[SYNC] Large move detected. Slicing into {num_slices} synchronized steps of max {MAX_STEP_UNITS} PU.")

            # Calculate the fractional step for each slice
            step_head = total_dist_head / num_slices
            step_link2 = total_dist_link2 / num_slices
            step_q1 = total_q1 / num_slices
            step_q2 = total_q2 / num_slices
            step_q3 = total_q3 / num_slices

            # --- 2. EXECUTE SLICES ---
            for i in range(num_slices):
                duration_ms = 1000
                payload = struct.pack('<H 8f', duration_ms, 
                                      step_head, step_link2, 
                                      step_q1, step_q2, step_q3, 
                                      0.0, 0.0, 0.0)
                
                checksum = calculate_checksum(payload)
                packet = bytes([SYNC_BYTE_1, SYNC_BYTE_2]) + payload + bytes([checksum])
                
                ser.write(packet)
                ser.flush()
                print(f"  -> Sending Slice {i+1}/{num_slices} ... ", end="", flush=True)
                
                # Wait for STM32 to finish this specific chunk
                while True:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if "REQ" in line:
                        print("Done.")
                        break
                    
            print("[TX] Full command execution complete!")
                    
        except ValueError:
            print("[ERROR] Invalid input. Numbers only.\n")
        except KeyboardInterrupt:
            break

    # Send sleep command
    print("\n[INFO] Putting STM32 to sleep...")
    dummy_payload = struct.pack('<H 8f', 65535, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    try:
        ser.write(bytes([SYNC_BYTE_1, SYNC_BYTE_2]) + dummy_payload + bytes([calculate_checksum(dummy_payload)]))
        ser.close()
    except: pass

if __name__ == "__main__":
    main()