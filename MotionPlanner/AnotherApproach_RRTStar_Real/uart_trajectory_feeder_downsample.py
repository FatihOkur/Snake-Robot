import serial
import json
import time
import struct
import sys

# --- CONFIGURATION ---
JSON_FILE = 'robot_path_commands.json'
UART_PORT = '/dev/ttyAMA0'
BAUD_RATE = 115200

SYNC_BYTE_1 = 0xAA
SYNC_BYTE_2 = 0xBB

# --- DOWNSAMPLING CONFIG ---
# A step is only SENT when the servo angle has changed by at least this much
# (in degrees) relative to the LAST SENT step. Smaller intermediate steps are
# MERGED into the next sent step: their DC distances are summed (so the robot
# still travels the full path length), and the servo angle used is the actual
# planned value at the moment the threshold is crossed (angles are NOT scaled
# or invented -- only the path is re-sampled more coarsely).
#
# Set this to just ABOVE your servo's mechanical deadband. Typical hobby servo
# deadband is ~1-2 deg. If servos still don't move on small steps, raise it.
SERVO_DEADBAND_DEG = 2.0

def calculate_checksum(payload_bytes):
    return sum(payload_bytes) & 0xFF

def get_yaw(cmd):
    s = cmd["servo_yaw_commands"]
    return (float(s["q1_deg"]), float(s["q2_deg"]), float(s["q3_deg"]))

def get_dist(cmd):
    d = cmd["dc_motor_commands"]
    return (float(d["segment1_head_distance_units"]),
            float(d["segment3_link2_distance_units"]))

def main():
    # 1. Load the Trajectory
    try:
        with open(JSON_FILE, 'r') as f:
            raw_trajectory = json.load(f)
        print(f"[INFO] Loaded {len(raw_trajectory)} raw steps from {JSON_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not load JSON: {e}")
        sys.exit(1)

    # 1.5 DOWNSAMPLE / MERGE PASS
    # Walk the raw steps. Accumulate DC distance every step. Only emit a merged
    # step when the servo yaw has moved past the deadband from the last EMITTED
    # step (or on the final step). The emitted step carries:
    #   - the SUMMED DC distances of all steps merged into it (path preserved)
    #   - the ACTUAL planned yaw angles of the current step (not modified)
    #   - the duration of the current step
    trajectory = []
    acc_head = 0.0
    acc_link2 = 0.0
    last_sent_yaw = None
    n = len(raw_trajectory)

    for i, cmd in enumerate(raw_trajectory):
        dh, dl = get_dist(cmd)
        acc_head += dh
        acc_link2 += dl

        q1, q2, q3 = get_yaw(cmd)

        if last_sent_yaw is None:
            angle_moved = True  # always emit the first step
        else:
            dq1 = abs(q1 - last_sent_yaw[0])
            dq2 = abs(q2 - last_sent_yaw[1])
            dq3 = abs(q3 - last_sent_yaw[2])
            angle_moved = (dq1 >= SERVO_DEADBAND_DEG or
                           dq2 >= SERVO_DEADBAND_DEG or
                           dq3 >= SERVO_DEADBAND_DEG)

        is_last = (i == n - 1)

        # Also emit if there is meaningful DC motion piled up but no further
        # steps will come (last), or if angle threshold crossed.
        if angle_moved or is_last:
            merged = {
                "step_duration_ms": int(cmd["step_duration_ms"]),
                "dc_motor_commands": {
                    "segment1_head_distance_units": acc_head,
                    "segment3_link2_distance_units": acc_link2,
                },
                "servo_yaw_commands": {
                    "q1_deg": q1, "q2_deg": q2, "q3_deg": q3,
                },
                "servo_pitch_commands": cmd["servo_pitch_commands"],
            }
            trajectory.append(merged)
            last_sent_yaw = (q1, q2, q3)
            acc_head = 0.0
            acc_link2 = 0.0

    total_steps = len(trajectory)
    print(f"[INFO] Downsampled {n} raw steps -> {total_steps} sent steps "
          f"(servo deadband {SERVO_DEADBAND_DEG} deg).")
    # Sanity: confirm total path length is preserved
    raw_sum_h = sum(get_dist(c)[0] for c in raw_trajectory)
    new_sum_h = sum(c["dc_motor_commands"]["segment1_head_distance_units"] for c in trajectory)
    print(f"[INFO] Head distance check: raw={raw_sum_h:.4f}  sent={new_sum_h:.4f} (should match).")

    # 2. Open UART Connection
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1, exclusive=True)
        print(f"[INFO] Connected to STM32 on {UART_PORT} at {BAUD_RATE} baud.")
        ser.reset_input_buffer()
        time.sleep(1)
    except Exception as e:
        print(f"[ERROR] Could not open UART. Port may be busy or disabled: {e}")
        sys.exit(1)

    step_index = 0
    print("[INFO] Waiting for STM32 to request the first step...")

    # 3. Main Communication Loop (identical protocol to the original _Real feeder)
    try:
        while step_index < total_steps:

            # KORUMA: I/O hatasında portu yeniden aç
            try:
                incoming_bytes = ser.readline()
            except Exception as e:
                print(f"\n[CRITICAL] Fiziksel kopma tespit edildi (I/O Hatasi): {e}")
                print(f"[CRITICAL] Port yeniden baslatiliyor...")
                try:
                    ser.close()
                except:
                    pass
                time.sleep(0.5)
                while True:
                    try:
                        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1, exclusive=True)
                        ser.reset_input_buffer()
                        print("[BILGI] Port kurtarildi. Devam ediliyor...\n")
                        break
                    except Exception as err:
                        print(f"[HATA] Port hala mesgul. 1 sn sonra tekrar... ({err})")
                        time.sleep(1)
                continue

            if incoming_bytes:
                incoming_str = incoming_bytes.decode('utf-8', errors='ignore').strip()

                if "REQ" in incoming_str:
                    print(f"[TX] STM32 requested data. Sending Step {step_index + 1}/{total_steps}...")

                    cmd = trajectory[step_index]

                    duration_ms = int(cmd["step_duration_ms"])
                    dist_head = float(cmd["dc_motor_commands"]["segment1_head_distance_units"])
                    dist_link2 = float(cmd["dc_motor_commands"]["segment3_link2_distance_units"])

                    q1_yaw = float(cmd["servo_yaw_commands"]["q1_deg"])
                    q2_yaw = float(cmd["servo_yaw_commands"]["q2_deg"])
                    q3_yaw = float(cmd["servo_yaw_commands"]["q3_deg"])

                    q1_pitch = float(cmd["servo_pitch_commands"]["q1_pitch_deg"])
                    q2_pitch = float(cmd["servo_pitch_commands"]["q2_pitch_deg"])
                    q3_pitch = float(cmd["servo_pitch_commands"]["q3_pitch_deg"])

                    payload = struct.pack('<H 8f',
                                          duration_ms,
                                          dist_head, dist_link2,
                                          q1_yaw, q2_yaw, q3_yaw,
                                          q1_pitch, q2_pitch, q3_pitch)

                    checksum = calculate_checksum(payload)
                    packet = bytes([SYNC_BYTE_1, SYNC_BYTE_2]) + payload + bytes([checksum])

                    try:
                        ser.write(packet)
                        ser.flush()
                        step_index += 1
                        time.sleep(0.01)
                        ser.reset_input_buffer()
                    except Exception as e:
                        print(f"[HATA] Yazma esnasinda port koptu! Kurtarma dongusune girilecek: {e}")

                elif incoming_str:
                    print(f"[STM32 DEBUG] {incoming_str}")

        print("[INFO] Trajectory complete. Sending END dummy packet.")
        dummy_payload = struct.pack('<H 8f', 65535, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        dummy_checksum = calculate_checksum(dummy_payload)
        dummy_packet = bytes([SYNC_BYTE_1, SYNC_BYTE_2]) + dummy_payload + bytes([dummy_checksum])
        try:
            ser.write(dummy_packet)
            ser.flush()
        except:
            pass
        ser.close()

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Closing port.")
        try:
            ser.close()
        except:
            pass

if __name__ == "__main__":
    main()