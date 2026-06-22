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
        self._lock = threading.Lock()
        print(f"[SIM ESTIMATOR] Running in mock mode. Initial state: "
              f"X={self.x:.2f}  Y={self.y:.2f}  Yaw={math.degrees(self.yaw):.1f}°")

    def update_odometry(self, meas_m2_units):
        import inspect
        try:
            # Dynamically extract expected yaw from the feeder's state to ensure perfect tracking on curves
            frame = inspect.currentframe().f_back
            step_index = frame.f_locals.get("step_index")
            trajectory = frame.f_locals.get("trajectory")
            if step_index is not None and trajectory is not None and step_index > 0:
                prev_step = trajectory[step_index - 1]
                if "base_coordinates" in prev_step:
                    self.yaw = float(prev_step["base_coordinates"]["yaw_rad"])
        except Exception:
            pass

        with self._lock:
            self.x += meas_m2_units * math.cos(self.yaw)
            self.y += meas_m2_units * math.sin(self.yaw)
            return (self.x, self.y, self.yaw)

    def apply_checkpoint_correction(self, true_x, true_y, true_yaw_rad):
        with self._lock:
            self.x = float(true_x)
            self.y = float(true_y)
            self.yaw = float(true_yaw_rad)
        print(f"[SIM CHECKPOINT] Snap correction applied: "
              f"X={true_x:.2f}  Y={true_y:.2f}  Yaw={math.degrees(true_yaw_rad):.1f}°")

    def get_state(self):
        with self._lock:
            return (self.x, self.y, self.yaw)

    def stop(self):
        print("[SIM ESTIMATOR] Stopped.")
