import math
import threading

class StateEstimator:
    """
    Simulated StateEstimator stub. It mimics the public interface of the
    real StateEstimator but performs simple unicycle dead reckoning
    without attempting to open I2C/SMBus (which crashes on Windows).
    """

    def __init__(self, start_x, start_y, start_yaw_rad):
        self.x = float(start_x)
        self.y = float(start_y)
        self.yaw = float(start_yaw_rad)
        self._yaw_drift = 0.0
        self._lock = threading.Lock()
        print(f"[SIM ESTIMATOR] Running in mock mode. Initial state: "
              f"X={self.x:.2f}  Y={self.y:.2f}  Yaw={math.degrees(self.yaw):.1f}°")

    def update_odometry(self, meas_m2_units, planned_yaw_rad=None):
        import random
        with self._lock:
            if planned_yaw_rad is not None:
                self._yaw_drift += random.uniform(-0.002, 0.002)
                # Keep it bounded
                self._yaw_drift = max(-0.1, min(0.1, self._yaw_drift))
                self.yaw = planned_yaw_rad + self._yaw_drift

            self.x += meas_m2_units * math.cos(self.yaw)
            self.y += meas_m2_units * math.sin(self.yaw)
            return (self.x, self.y, self.yaw)

    def apply_checkpoint_correction(self, true_x, true_y, true_yaw_rad):
        with self._lock:
            self.x = float(true_x)
            self.y = float(true_y)
            self.yaw = float(true_yaw_rad)
            self._yaw_drift = 0.0
        print(f"[SIM CHECKPOINT] Snap correction applied: "
              f"X={true_x:.2f}  Y={true_y:.2f}  Yaw={math.degrees(true_yaw_rad):.1f}°")

    def get_state(self):
        with self._lock:
            return (self.x, self.y, self.yaw)

    def stop(self):
        print("[SIM ESTIMATOR] Stopped.")
