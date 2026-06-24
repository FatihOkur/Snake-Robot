import serial
import json
import time
import struct
import sys
import re
import math
import subprocess
import os
import threading
import socket

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────
JSON_FILE = 'robot_path_commands.json'
REPLAN_OUTPUT_FILE = 'replanned_path.json'
UART_PORT = '/dev/ttyAMA0'
BAUD_RATE = 115200

SYNC_BYTE_1 = 0xAA
SYNC_BYTE_2 = 0xBB

# --- PHYSICALLY CALIBRATED unit conversion constant ---
UNITS_TO_REV = 0.61224

# --- DOWNSAMPLING CONFIG ---
SERVO_DEADBAND_DEG = 2.0
MAX_DC_UNITS_PER_STEP = 2.0

# --- DYNAMIC REPLANNING ---
REPLAN_THRESHOLD = 5.0          # map units of position error before replan
PLANNER_SCRIPT   = 'main.py'   # RRT* planner entry point


# ──────────────────────────────────────────────────────────────────
# PURE VISION STATE ESTIMATOR  (replaces encoder dead-reckoning)
# ──────────────────────────────────────────────────────────────────
class PureVisionStateEstimator:
    """
    100 % camera-based localisation.
    Listens on a UDP socket for JSON packets of the form:
        {"x": float, "y": float, "yaw_rad": float}
    broadcast by the overhead ArUco checkpoint vision system.
    Every received packet overwrites the current state instantly.
    """

    def __init__(self, start_x, start_y, start_yaw):
        self.x = start_x
        self.y = start_y
        self.yaw = start_yaw
        self.lock = threading.Lock()
        self.new_data_received = False

        # --- UDP Listener for Camera Checkpoints ---
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 5005))
        self.sock.settimeout(0.5)
        self.running = True

        self.thread = threading.Thread(target=self._listen_udp, daemon=True)
        self.thread.start()

    def _listen_udp(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
                msg = json.loads(data.decode("utf-8"))
                with self.lock:
                    # 100% Trust the camera! Overwrite coordinates instantly.
                    self.x = msg["x"]
                    self.y = msg["y"]
                    self.yaw = msg["yaw_rad"]
                    self.new_data_received = True
            except socket.timeout:
                continue
            except Exception:
                pass

    def get_state(self):
        with self.lock:
            return self.x, self.y, self.yaw

    def get_state_and_clear_flag(self):
        with self.lock:
            fresh = self.new_data_received
            self.new_data_received = False
            return self.x, self.y, self.yaw, fresh

    def stop(self):
        self.running = False
        self.thread.join()


# ──────────────────────────────────────────────────────────────────
# [DEPRECATED] INVERSE KINEMATICS — Nose ➜ Seg3 (Anchor / J3)
# The planner now uses a Head-anchored state vector, so the
# Nose→J3 conversion is no longer needed for replanning.
# Kept for reference only.
# ──────────────────────────────────────────────────────────────────
# Segment lengths in map units  (1 unit = 10 cm)
L_HEAD = 2.5     # Nose  → J1
L_SEG2 = 3.05    # J1    → J2
L_SEG3 = 3.65    # J2    → J3


def calculate_j3_from_nose(nose_x, nose_y, nose_yaw_rad, q1_deg, q2_deg):
    """
    [DEPRECATED] Walk backwards along the robot's kinematic chain from the
    physical Nose to the Segment-3 anchor point (J3).
    No longer used for replanning — the planner now accepts head coordinates
    directly. Kept for reference.

    Convention:  +angle = Left (CCW),  –angle = Right (CW).
    The joint angle is subtracted because rotating the joint LEFT
    swings the segment behind it to the RIGHT in global frame.

    Parameters
    ----------
    nose_x, nose_y : float   – Nose position in map units.
    nose_yaw_rad   : float   – Nose heading (radians, math convention).
    q1_deg         : float   – Joint-1 yaw command (degrees).
    q2_deg         : float   – Joint-2 yaw command (degrees).

    Returns
    -------
    (j3_x, j3_y, yaw_seg3)  – Anchor position and heading.
    """
    yaw_head = nose_yaw_rad
    yaw_seg2 = yaw_head - math.radians(q1_deg)
    yaw_seg3 = yaw_seg2 - math.radians(q2_deg)

    # Nose → J1 (walk back along Head segment)
    j1_x = nose_x - L_HEAD * math.cos(yaw_head)
    j1_y = nose_y - L_HEAD * math.sin(yaw_head)

    # J1 → J2 (walk back along Seg2)
    j2_x = j1_x - L_SEG2 * math.cos(yaw_seg2)
    j2_y = j1_y - L_SEG2 * math.sin(yaw_seg2)

    # J2 → J3 / Anchor (walk back along Seg3)
    j3_x = j2_x - L_SEG3 * math.cos(yaw_seg3)
    j3_y = j2_y - L_SEG3 * math.sin(yaw_seg3)

    return j3_x, j3_y, yaw_seg3


# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────
def calculate_checksum(payload_bytes):
    return sum(payload_bytes) & 0xFF


def get_yaw(cmd):
    s = cmd["servo_yaw_commands"]
    return (float(s["q1_deg"]), float(s["q2_deg"]), float(s["q3_deg"]))


def get_dist(cmd):
    d = cmd["dc_motor_commands"]
    return (float(d["segment1_head_distance_units"]),
            float(d["segment3_link2_distance_units"]))


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
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
    trajectory = []
    acc_duration = 0
    acc_head = 0.0
    acc_link2 = 0.0
    last_sent_yaw = None
    n = len(raw_trajectory)

    def emit(idx_cmd, ad, ah, al, q):
        entry = {
            "step_duration_ms": ad,
            "dc_motor_commands": {
                "segment1_head_distance_units": ah,
                "segment3_link2_distance_units": al,
            },
            "servo_yaw_commands": {
                "q1_deg": q[0], "q2_deg": q[1], "q3_deg": q[2],
            },
            "servo_pitch_commands": idx_cmd["servo_pitch_commands"],
        }
        # Propagate base_coordinates from the raw command (needed for
        # the replanning trigger to compare expected vs estimated position)
        if "base_coordinates" in idx_cmd:
            entry["base_coordinates"] = idx_cmd["base_coordinates"]
        trajectory.append(entry)

    prev_cmd = None

    for i, cmd in enumerate(raw_trajectory):
        dh, dl = get_dist(cmd)

        # --- DIRECTION-CHANGE FLUSH ---
        head_reversal  = (acc_head  * dh < -1e-12)
        link2_reversal = (acc_link2 * dl < -1e-12)
        if (head_reversal or link2_reversal) and prev_cmd is not None:
            pq = get_yaw(prev_cmd)
            emit(prev_cmd, acc_duration, acc_head, acc_link2, pq)
            last_sent_yaw = pq
            acc_duration = 0
            acc_head = 0.0
            acc_link2 = 0.0

        acc_duration += int(cmd["step_duration_ms"])
        acc_head += dh
        acc_link2 += dl

        q1, q2, q3 = get_yaw(cmd)

        if last_sent_yaw is None:
            angle_moved = True
        else:
            dq1 = abs(q1 - last_sent_yaw[0])
            dq2 = abs(q2 - last_sent_yaw[1])
            dq3 = abs(q3 - last_sent_yaw[2])
            angle_moved = (dq1 >= SERVO_DEADBAND_DEG or
                           dq2 >= SERVO_DEADBAND_DEG or
                           dq3 >= SERVO_DEADBAND_DEG)

        is_last = (i == n - 1)

        dc_cap_hit = (abs(acc_head) >= MAX_DC_UNITS_PER_STEP or
                      abs(acc_link2) >= MAX_DC_UNITS_PER_STEP)

        if angle_moved or dc_cap_hit or is_last:
            emit(cmd, acc_duration, acc_head, acc_link2, (q1, q2, q3))
            last_sent_yaw = (q1, q2, q3)
            acc_duration = 0
            acc_head = 0.0
            acc_link2 = 0.0

        prev_cmd = cmd

    total_steps = len(trajectory)
    print(f"[INFO] Downsampled {n} raw steps -> {total_steps} sent steps "
          f"(servo deadband {SERVO_DEADBAND_DEG} deg).")
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

    # 2.5 Initialize the State Estimator (Pure Vision — UDP camera)
    START_X = float(trajectory[0]["base_coordinates"]["x"])
    START_Y = float(trajectory[0]["base_coordinates"]["y"])
    START_YAW = float(trajectory[0]["base_coordinates"]["yaw_rad"])
    estimator = PureVisionStateEstimator(START_X, START_Y, START_YAW)

    step_index = 0
    print("[INFO] Waiting for STM32 to request the first step...")

    # 3. Main Communication Loop
    try:
        while step_index < total_steps:

            # KORUMA: I/O hatasinda portu yeniden ac
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
                    # ── Vision-based position check (runs every REQ after
                    #    the first step has been sent) ──
                    if step_index > 0:
                        est_x, est_y, est_yaw, is_fresh = estimator.get_state_and_clear_flag()
                        print(f"[VISION] Map Position: "
                              f"X={est_x:.3f}  Y={est_y:.3f}  "
                              f"Yaw={math.degrees(est_yaw):.1f}°")

                        if is_fresh:
                            # --- DYNAMIC REPLANNING TRIGGER ---
                            # The planner now uses a Head-anchored state vector.
                            # The vision estimator reports the *Nose/Head* pose
                            # directly (est_x, est_y, est_yaw), and
                            # base_coordinates in the JSON also track the Head.
                            # No Nose→J3 conversion needed.
                            #
                            # ── SENSOR HEADING FRAME ──
                            # The ArUco marker is on the Head, so est_yaw IS
                            # yaw_head. If the IMU (mounted on Seg3) were used
                            # instead, you would need:
                            #   yaw_head = yaw_seg3 + radians(q1 + q2)
                            # Currently using vision, so this is identity:
                            head_x = est_x
                            head_y = est_y
                            head_yaw = est_yaw  # Already yaw_head from ArUco

                            prev_step = trajectory[step_index - 1]
                            if "base_coordinates" in prev_step:
                                exp_x = prev_step["base_coordinates"]["x"]
                                exp_y = prev_step["base_coordinates"]["y"]
                                pos_error = math.hypot(head_x - exp_x, head_y - exp_y)
                                print(f"[NAV] Head est: ({head_x:.2f}, {head_y:.2f})  "
                                      f"Expected: ({exp_x:.2f}, {exp_y:.2f})  "
                                      f"Error: {pos_error:.3f} units")

                                if pos_error > REPLAN_THRESHOLD:
                                    print(f"[REPLAN] Off course! Error = "
                                          f"{pos_error:.3f} units. "
                                          f"Recalculating path...")

                                    # Grab current joint states from the last sent command
                                    cur_q1 = float(prev_step["servo_yaw_commands"]["q1_deg"])
                                    cur_q2 = float(prev_step["servo_yaw_commands"]["q2_deg"])
                                    cur_q3 = float(prev_step["servo_yaw_commands"]["q3_deg"])

                                    script_dir = os.path.dirname(os.path.abspath(__file__))
                                    planner_path = os.path.join(script_dir, PLANNER_SCRIPT)

                                    # Pass head pose directly — no J3 conversion
                                    replan_cmd = [
                                        sys.executable, planner_path,
                                        '--start_x',       str(round(head_x, 4)),
                                        '--start_y',       str(round(head_y, 4)),
                                        '--start_yaw_rad', str(round(head_yaw, 6)),
                                        '--start_q1',      str(cur_q1),
                                        '--start_q2',      str(cur_q2),
                                        '--start_q3',      str(cur_q3),
                                        '--out',           REPLAN_OUTPUT_FILE,
                                    ]
                                    print(f"[REPLAN] Invoking: {' '.join(replan_cmd)}")
                                    result = subprocess.run(
                                        replan_cmd,
                                        cwd=script_dir,
                                        capture_output=True, text=True, timeout=300,
                                    )
                                    print(result.stdout)
                                    if result.returncode != 0:
                                        print(f"[REPLAN][ERROR] Planner failed:\n"
                                              f"{result.stderr}")
                                        print("[REPLAN] Continuing with OLD trajectory.")
                                    else:
                                        # Reload the freshly planned trajectory
                                        try:
                                            json_path = os.path.join(script_dir, REPLAN_OUTPUT_FILE)
                                            with open(json_path, 'r') as jf:
                                                new_raw = json.load(jf)
                                            # Re-run the downsample pass
                                            trajectory.clear()
                                            acc_duration = 0
                                            acc_head = 0.0
                                            acc_link2 = 0.0
                                            last_sent_yaw = None
                                            n = len(new_raw)
                                            prev_cmd = None
                                            for ii, ccmd in enumerate(new_raw):
                                                dh, dl = get_dist(ccmd)
                                                hr = (acc_head * dh < -1e-12)
                                                lr = (acc_link2 * dl < -1e-12)
                                                if (hr or lr) and prev_cmd is not None:
                                                    pq = get_yaw(prev_cmd)
                                                    emit(prev_cmd, acc_duration, acc_head, acc_link2, pq)
                                                    last_sent_yaw = pq
                                                    acc_duration = 0
                                                    acc_head = 0.0
                                                    acc_link2 = 0.0
                                                
                                                acc_duration += int(ccmd["step_duration_ms"])
                                                acc_head += dh
                                                acc_link2 += dl
                                                cq1, cq2, cq3 = get_yaw(ccmd)
                                                if last_sent_yaw is None:
                                                    ang_moved = True
                                                else:
                                                    ang_moved = (
                                                        abs(cq1 - last_sent_yaw[0]) >= SERVO_DEADBAND_DEG or
                                                        abs(cq2 - last_sent_yaw[1]) >= SERVO_DEADBAND_DEG or
                                                        abs(cq3 - last_sent_yaw[2]) >= SERVO_DEADBAND_DEG
                                                    )
                                                is_last = (ii == n - 1)
                                                dc_cap = (
                                                    abs(acc_head) >= MAX_DC_UNITS_PER_STEP or
                                                    abs(acc_link2) >= MAX_DC_UNITS_PER_STEP
                                                )
                                                if ang_moved or dc_cap or is_last:
                                                    emit(ccmd, acc_duration, acc_head, acc_link2, (cq1, cq2, cq3))
                                                    last_sent_yaw = (cq1, cq2, cq3)
                                                    acc_duration = 0
                                                    acc_head = 0.0
                                                    acc_link2 = 0.0
                                                prev_cmd = ccmd

                                            total_steps = len(trajectory)
                                            step_index = 0
                                            print(f"[REPLAN] Loaded new trajectory: "
                                                  f"{total_steps} steps. Resuming.")
                                            # Fall through to send the new step 0 immediately
                                        except Exception as reload_err:
                                            print(f"[REPLAN][ERROR] Failed to reload "
                                                  f"JSON: {reload_err}")
                                            print("[REPLAN] Continuing with OLD "
                                                  "trajectory.")
                        else:
                            print(f"[NAV] Blind spot (no checkpoint). "
                                  f"Continuing open-loop to Step {step_index}")

                    if step_index >= total_steps:
                        break

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
                        ser.reset_input_buffer()  # Drop stale REQs; wait for a fresh one
                    except Exception as e:
                        print(f"[HATA] Yazma esnasinda port koptu! Kurtarma dongusune girilecek: {e}")

                elif incoming_str:
                    print(f"[STM32 DEBUG] {incoming_str}")

        # --- After all steps sent, wait for the final REQ as confirmation ---
        print("[INFO] Last step sent. Waiting for final STM32 acknowledgement...")
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
            except Exception:
                break
            if "REQ" in line:
                est_x, est_y, est_yaw = estimator.get_state()
                print(f"[VISION] Final Map Position: "
                      f"X={est_x:.3f}  Y={est_y:.3f}  "
                      f"Yaw={math.degrees(est_yaw):.1f}°")
                break

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
        estimator.stop()

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Closing port.")
        try:
            estimator.stop()
        except:
            pass
        try:
            ser.close()
        except:
            pass


if __name__ == "__main__":
    main()
