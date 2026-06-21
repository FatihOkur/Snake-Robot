import serial
import json
import time
import struct
import sys
import re
import math
import subprocess
import os

from state_estimator import StateEstimator

# --- CONFIGURATION ---
JSON_FILE = 'robot_path_commands.json'
UART_PORT = '/dev/ttyAMA0'
BAUD_RATE = 115200

SYNC_BYTE_1 = 0xAA
SYNC_BYTE_2 = 0xBB

# --- ENCODER -> UNITS CONVERSION (must match STM32 firmware) ---
PULSES_PER_REV = 7392.0
UNITS_TO_REV   = 1.0   # placeholder, identical to the STM32 #define

# --- DOWNSAMPLING CONFIG ---
# A step is only SENT when the servo angle has changed by at least this much
# (in degrees) relative to the LAST SENT step. Smaller intermediate steps are
# MERGED into the next sent step: their DC distances are summed (so the robot
# still travels the full path length), and the servo angle used is the actual
# planned value at the moment the threshold is crossed (angles are NOT scaled
# or invented -- only the path is re-sampled more coarsely).
SERVO_DEADBAND_DEG = 2.0

# Maximum DC distance (in planner units) allowed to accumulate into a single
# sent step. Splits long straights into several moderate moves WITHOUT touching
# angles. Tune ~1.5-2.5 units.
MAX_DC_UNITS_PER_STEP = 2.0

# --- SENSOR FUSION: map starting pose (hardcoded for current deployment) ---
START_X   = 17.0
START_Y   = 10.0
START_YAW = math.radians(90)          # facing +y = North

# --- DYNAMIC REPLANNING ---
REPLAN_THRESHOLD = 0.5                 # map units of position error before replan
PLANNER_SCRIPT   = 'main.py'           # RRT* planner entry point


def calculate_checksum(payload_bytes):
    return sum(payload_bytes) & 0xFF


def get_yaw(cmd):
    s = cmd["servo_yaw_commands"]
    return (float(s["q1_deg"]), float(s["q2_deg"]), float(s["q3_deg"]))


def get_dist(cmd):
    d = cmd["dc_motor_commands"]
    return (float(d["segment1_head_distance_units"]),
            float(d["segment3_link2_distance_units"]))


# --- STEP 3: parse the measured encoder deltas embedded in the REQ line ---
_REQ_M1 = re.compile(r'M1:(-?\d+)')
_REQ_M2 = re.compile(r'M2:(-?\d+)')


def parse_req_deltas(line):
    """
    From 'REQ M1:<pulses> M2:<pulses>' return (m1_pulses, m2_pulses) ints,
    or (None, None) if not parseable. Robust to extra whitespace/garbage.
    """
    m1 = _REQ_M1.search(line)
    m2 = _REQ_M2.search(line)
    if m1 and m2:
        return int(m1.group(1)), int(m2.group(1))
    return None, None


def pulses_to_units(pulses):
    """Encoder pulses -> planner distance units (must mirror the STM32 math)."""
    return (pulses / PULSES_PER_REV) / UNITS_TO_REV


def main():
    # 1. Load the Trajectory
    try:
        with open(JSON_FILE, 'r') as f:
            raw_trajectory = json.load(f)
        print(f"[INFO] Loaded {len(raw_trajectory)} raw steps from {JSON_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not load JSON: {e}")
        sys.exit(1)

    # 1.5 DOWNSAMPLE / MERGE PASS (unchanged from your version)
    trajectory = []
    acc_head = 0.0
    acc_link2 = 0.0
    last_sent_yaw = None
    n = len(raw_trajectory)

    def emit(idx_cmd, ah, al, q):
        entry = {
            "step_duration_ms": int(idx_cmd["step_duration_ms"]),
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
            emit(prev_cmd, acc_head, acc_link2, pq)
            last_sent_yaw = pq
            acc_head = 0.0
            acc_link2 = 0.0

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
            emit(cmd, acc_head, acc_link2, (q1, q2, q3))
            last_sent_yaw = (q1, q2, q3)
            acc_head = 0.0
            acc_link2 = 0.0

        prev_cmd = cmd

    total_steps = len(trajectory)
    print(f"[INFO] Downsampled {n} raw steps -> {total_steps} sent steps "
          f"(servo deadband {SERVO_DEADBAND_DEG} deg).")
    raw_sum_h = sum(get_dist(c)[0] for c in raw_trajectory)
    new_sum_h = sum(c["dc_motor_commands"]["segment1_head_distance_units"] for c in trajectory)
    print(f"[INFO] Head distance check: raw={raw_sum_h:.4f}  sent={new_sum_h:.4f} (should match).")

    # --- STEP 3: measured-odometry log, one row per completed step ---
    # measured_odom[k] = {'cmd_m2', 'meas_m2', 'cmd_m1', 'meas_m1'} for step k.
    # This is the data Step 4 (dead-reckoning) will consume instead of commands.
    measured_odom = []

    # 2. Open UART Connection
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1, exclusive=True)
        print(f"[INFO] Connected to STM32 on {UART_PORT} at {BAUD_RATE} baud.")
        ser.reset_input_buffer()
        time.sleep(1)
    except Exception as e:
        print(f"[ERROR] Could not open UART. Port may be busy or disabled: {e}")
        sys.exit(1)

    # 2.5 Initialize the State Estimator (Sensor Fusion Engine)
    estimator = StateEstimator(START_X, START_Y, START_YAW)

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
                    # --- STEP 3: read MEASURED deltas of the step that just ran ---
                    # The delta in THIS REQ belongs to the PREVIOUS step we sent
                    # (step_index-1). The very first REQ (step_index==0) carries
                    # 0/0 and is skipped.
                    m1_pulses, m2_pulses = parse_req_deltas(incoming_str)
                    if m1_pulses is not None and step_index > 0:
                        meas_m2 = pulses_to_units(m2_pulses)
                        meas_m1 = pulses_to_units(m1_pulses)
                        prev = trajectory[step_index - 1]["dc_motor_commands"]
                        cmd_m2 = float(prev["segment3_link2_distance_units"])
                        cmd_m1 = float(prev["segment1_head_distance_units"])

                        measured_odom.append({
                            "step": step_index - 1,
                            "cmd_m2": cmd_m2, "meas_m2": meas_m2,
                            "cmd_m1": cmd_m1, "meas_m1": meas_m1,
                        })

                        slip = cmd_m2 - meas_m2
                        print(f"[ODOM] step {step_index-1:3d} | "
                              f"M2 cmd={cmd_m2:+.4f} meas={meas_m2:+.4f} "
                              f"slip={slip:+.4f} | "
                              f"M1 cmd={cmd_m1:+.4f} meas={meas_m1:+.4f}")

                        # --- SENSOR FUSION (Task 2): dead-reckon using meas_m2 ---
                        est_x, est_y, est_yaw = estimator.update_odometry(meas_m2)
                        print(f"[FUSION] Map Position: "
                              f"X={est_x:.3f}  Y={est_y:.3f}  "
                              f"Yaw={math.degrees(est_yaw):.1f}°")

                        # --- DYNAMIC REPLANNING TRIGGER ---
                        prev_step = trajectory[step_index - 1]
                        if "base_coordinates" in prev_step:
                            exp_x = prev_step["base_coordinates"]["x"]
                            exp_y = prev_step["base_coordinates"]["y"]
                            pos_error = math.hypot(est_x - exp_x, est_y - exp_y)
                            print(f"[NAV] Position error: {pos_error:.3f} units")

                            if pos_error > REPLAN_THRESHOLD:
                                print(f"[REPLAN] Off course! Error = "
                                      f"{pos_error:.3f} units. "
                                      f"Recalculating path...")

                                # Use last_sent_yaw for the physical joint state
                                # (servos are deterministic, so the last command
                                # we sent is the current physical configuration)
                                cur_q1 = float(last_sent_yaw[0]) if last_sent_yaw else 0.0
                                cur_q2 = float(last_sent_yaw[1]) if last_sent_yaw else 0.0
                                cur_q3 = float(last_sent_yaw[2]) if last_sent_yaw else 0.0

                                script_dir = os.path.dirname(os.path.abspath(__file__))
                                planner_path = os.path.join(script_dir, PLANNER_SCRIPT)

                                replan_cmd = [
                                    sys.executable, planner_path,
                                    '--start_x',       str(est_x),
                                    '--start_y',       str(est_y),
                                    '--start_yaw_rad', str(est_yaw),
                                    '--start_q1',      str(cur_q1),
                                    '--start_q2',      str(cur_q2),
                                    '--start_q3',      str(cur_q3),
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
                                        json_path = os.path.join(script_dir, JSON_FILE)
                                        with open(json_path, 'r') as jf:
                                            new_raw = json.load(jf)
                                        # Re-run the downsample pass
                                        trajectory.clear()
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
                                                emit(prev_cmd, acc_head, acc_link2, pq)
                                                last_sent_yaw = pq
                                                acc_head = 0.0
                                                acc_link2 = 0.0
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
                                                emit(ccmd, acc_head, acc_link2, (cq1, cq2, cq3))
                                                last_sent_yaw = (cq1, cq2, cq3)
                                                acc_head = 0.0
                                                acc_link2 = 0.0
                                            prev_cmd = ccmd

                                        total_steps = len(trajectory)
                                        step_index = 0
                                        measured_odom.clear()
                                        print(f"[REPLAN] Loaded new trajectory: "
                                              f"{total_steps} steps. Resuming.")
                                        continue  # immediately serve new step 0
                                    except Exception as reload_err:
                                        print(f"[REPLAN][ERROR] Failed to reload "
                                              f"JSON: {reload_err}")
                                        print("[REPLAN] Continuing with OLD "
                                              "trajectory.")

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
                        ser.reset_input_buffer()
                    except Exception as e:
                        print(f"[HATA] Yazma esnasinda port koptu! Kurtarma dongusune girilecek: {e}")

                elif incoming_str:
                    print(f"[STM32 DEBUG] {incoming_str}")

        # --- STEP 3: collect the final step's measured delta ---
        # After the last step is sent, the STM32 will emit one more REQ carrying
        # that step's measured delta. Read it so the odometry log is complete,
        # then send the END sentinel.
        print("[INFO] Last step sent. Waiting for its measured delta...")
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
            except Exception:
                break
            if "REQ" in line:
                m1_pulses, m2_pulses = parse_req_deltas(line)
                if m1_pulses is not None:
                    meas_m2 = pulses_to_units(m2_pulses)
                    meas_m1 = pulses_to_units(m1_pulses)
                    prev = trajectory[total_steps - 1]["dc_motor_commands"]
                    cmd_m2 = float(prev["segment3_link2_distance_units"])
                    cmd_m1 = float(prev["segment1_head_distance_units"])
                    measured_odom.append({
                        "step": total_steps - 1,
                        "cmd_m2": cmd_m2, "meas_m2": meas_m2,
                        "cmd_m1": cmd_m1, "meas_m1": meas_m1,
                    })
                    print(f"[ODOM] step {total_steps-1:3d} | "
                          f"M2 cmd={cmd_m2:+.4f} meas={meas_m2:+.4f} "
                          f"slip={cmd_m2-meas_m2:+.4f}")
                    # Final fusion update
                    est_x, est_y, est_yaw = estimator.update_odometry(meas_m2)
                    print(f"[FUSION] Final Map Position: "
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

        # --- STEP 3: persist the measured odometry for inspection / Step 4 ---
        try:
            with open("measured_odometry.json", "w") as f:
                json.dump(measured_odom, f, indent=2)
            tot_cmd = sum(r["cmd_m2"] for r in measured_odom)
            tot_meas = sum(r["meas_m2"] for r in measured_odom)
            print(f"[ODOM] Saved {len(measured_odom)} rows to measured_odometry.json")
            print(f"[ODOM] Total M2: commanded={tot_cmd:.4f}  measured={tot_meas:.4f}  "
                  f"({100.0*tot_meas/tot_cmd if tot_cmd else 0:.1f}% of commanded)")
        except Exception as e:
            print(f"[WARN] Could not save odometry log: {e}")

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
