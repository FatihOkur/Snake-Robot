#!/usr/bin/env python3
"""
State Estimator — Sensor Fusion Engine for the Snake Robot (Dead Reckoning).

Combines:
  - IMU gyro-integrated heading from YawTracker (mpu6050_yaw_step2.py)
  - Encoder odometry from Segment 3 (meas_m2) — the mechanical anchor

Coordinate convention (standard math):
  +x = 0 rad,  +y = π/2 rad

Only meas_m2 is used for (X, Y) projection.  meas_m1 is NEVER consumed here
because Segment 1 (Head) travels on a different turning radius due to q2
steering.  Segment 3 is the fixed wheelbase anchor.

Also hosts a non-blocking UDP listener (Task 4) that receives absolute map
coordinates from the ArUco checkpoint script (threeD_measure.py) and snaps
the dead-reckoned state back to reality.
"""

import math
import time
import json
import socket
import threading
from smbus2 import SMBus
from mpu6050_yaw_step2 import YawTracker, I2C_BUS

# --- UDP Checkpoint IPC ---
CHECKPOINT_UDP_IP   = "127.0.0.1"
CHECKPOINT_UDP_PORT = 5005


class StateEstimator:
    """
    Fuses IMU heading with Segment 3 encoder odometry to maintain a
    real-time (x, y, yaw) estimate of the robot's anchor on the map.
    """

    def __init__(self, start_x, start_y, start_yaw_rad):
        # --- Position state ---
        self.x = float(start_x)
        self.y = float(start_y)
        self._lock = threading.Lock()

        # --- IMU heading ---
        self._bus = SMBus(I2C_BUS)
        self._yaw_tracker = YawTracker(self._bus)
        print("[ESTIMATOR] Calibrating IMU bias (hold still)...")
        self._yaw_tracker.calibrate()
        self._yaw_tracker.reset_yaw(start_yaw_rad)
        self._yaw_tracker.start()
        print(f"[ESTIMATOR] IMU running.  Initial state: "
              f"X={self.x:.2f}  Y={self.y:.2f}  "
              f"Yaw={math.degrees(start_yaw_rad):.1f}°")

        # --- Checkpoint UDP listener (Task 4) ---
        self._running = True
        self._ckpt_thread = threading.Thread(
            target=self._checkpoint_listener, daemon=True
        )
        self._ckpt_thread.start()

    # ------------------------------------------------------------------
    # Core Odometry
    # ------------------------------------------------------------------
    def update_odometry(self, meas_m2_units):
        """
        Unicycle kinematics update.

        Parameters
        ----------
        meas_m2_units : float
            Actual distance the Segment 3 track rolled (planner map units),
            as reported by the STM32 encoder.  Positive = forward along the
            robot's heading, negative = reverse.

        Returns
        -------
        (x, y, yaw) : tuple of floats
            Updated map position and heading (radians).
        """
        yaw = self._yaw_tracker.get_yaw()

        with self._lock:
            self.x += meas_m2_units * math.cos(yaw)
            self.y += meas_m2_units * math.sin(yaw)
            return (self.x, self.y, yaw)

    # ------------------------------------------------------------------
    # Checkpoint Correction
    # ------------------------------------------------------------------
    def apply_checkpoint_correction(self, true_x, true_y, true_yaw_rad):
        """
        Overwrites the dead-reckoned state with an absolute measurement
        (e.g. from an ArUco fiducial).
        """
        with self._lock:
            self.x = float(true_x)
            self.y = float(true_y)
        self._yaw_tracker.reset_yaw(true_yaw_rad)
        print(f"[CHECKPOINT] Snap correction applied: "
              f"X={true_x:.2f}  Y={true_y:.2f}  "
              f"Yaw={math.degrees(true_yaw_rad):.1f}°")

    # ------------------------------------------------------------------
    # Accessor
    # ------------------------------------------------------------------
    def get_state(self):
        """Returns the current (x, y, yaw) estimate."""
        yaw = self._yaw_tracker.get_yaw()
        with self._lock:
            return (self.x, self.y, yaw)

    # ------------------------------------------------------------------
    # UDP Checkpoint Listener (Task 4)
    # ------------------------------------------------------------------
    def _checkpoint_listener(self):
        """
        Non-blocking UDP listener.  Expects JSON datagrams of the form:
            {"x": float, "y": float, "yaw_rad": float}
        sent by threeD_measure.py whenever an ArUco marker yields an
        absolute map position.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((CHECKPOINT_UDP_IP, CHECKPOINT_UDP_PORT))
        sock.settimeout(0.1)          # 100 ms poll so we can check _running
        print(f"[ESTIMATOR] Checkpoint listener active on "
              f"UDP {CHECKPOINT_UDP_IP}:{CHECKPOINT_UDP_PORT}")

        while self._running:
            try:
                data, _ = sock.recvfrom(1024)
                msg = json.loads(data.decode("utf-8"))
                cx = float(msg["x"])
                cy = float(msg["y"])
                cyaw = float(msg["yaw_rad"])
                self.apply_checkpoint_correction(cx, cy, cyaw)
            except socket.timeout:
                continue
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"[CHECKPOINT][WARN] Bad datagram ignored: {e}")
            except Exception as e:
                if self._running:
                    print(f"[CHECKPOINT][ERR] Listener error: {e}")
                break

        sock.close()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def stop(self):
        """Safely shut down the IMU thread, checkpoint listener, and I2C."""
        self._running = False
        self._yaw_tracker.stop()
        try:
            self._bus.close()
        except Exception:
            pass
        # Give the checkpoint thread a moment to exit its poll
        if self._ckpt_thread.is_alive():
            self._ckpt_thread.join(timeout=0.5)
        print("[ESTIMATOR] Stopped.")


# ======================================================================
# Standalone test / verification
# ======================================================================
if __name__ == "__main__":
    est = StateEstimator(start_x=17.0, start_y=10.0,
                         start_yaw_rad=math.radians(90))
    print("\n[TEST] StateEstimator running.  Move the robot or send a UDP "
          "checkpoint.  Ctrl+C to stop.\n"
          "       UDP test:  echo '{\"x\":17.5,\"y\":11.0,\"yaw_rad\":1.57}' "
          "| nc -u 127.0.0.1 5005\n")
    try:
        while True:
            x, y, yaw = est.get_state()
            print(f"   X={x:+8.3f}  Y={y:+8.3f}  "
                  f"Yaw={math.degrees(yaw):+7.2f}°", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        est.stop()
        x, y, yaw = est.get_state()
        print(f"\n[INFO] Final state: X={x:.3f}  Y={y:.3f}  "
              f"Yaw={math.degrees(yaw):.2f}°")
