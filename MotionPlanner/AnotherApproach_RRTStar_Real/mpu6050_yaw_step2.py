#!/usr/bin/env python3
"""
Step 2: Gyro bias calibration + yaw integration for the snake robot base.

Builds on Step 1. Mount-specific facts established by the raw-read test:
  - IMU is on the Segment 3 face of joint 3  -> yaw_IMU = yaw_seg3 directly.
  - The board's X axis points UP (rest reading ax ~ +1g), so the YAW RATE
    is gx, NOT gz.
  - CCW rotation (viewed from above) gives +gx, which matches the planner's
    positive-yaw convention. => NO sign flip.

What this script does:
  1. Calibrates the gx bias: robot held DEAD STILL, average many samples.
  2. Integrates (gx - bias) over real elapsed time into yaw_seg3.
  3. Runs the sampling/integration in its own steady-rate thread (decoupled
     from any motion/step machinery), exposing a thread-safe get_yaw().

This is the high-rate heading source. Distance fusion (encoder) and checkpoint
snapping come in later steps and CONSUME get_yaw(); they are not here.

Verification built in:
  - Prints live yaw so you can do the protractor test (rotate a known angle,
    check it reads back) and the still-drift test (leave it, watch drift/min).
"""

import time
import math
import threading
from smbus2 import SMBus

# ---------------------------------------------------------------------------
# Constants (from Step 1)
# ---------------------------------------------------------------------------
I2C_BUS = 1
MPU_ADDR = 0x68

REG_PWR_MGMT_1   = 0x6B
REG_SMPLRT_DIV   = 0x19
REG_CONFIG       = 0x1A
REG_GYRO_CONFIG  = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_XOUT_H = 0x3B

GYRO_SENS = 131.0          # LSB per deg/s at +/-250 dps

# --- Mount-specific yaw configuration ---
# gx is the yaw axis; CCW is positive => sign +1 (no flip).
YAW_SIGN = +1.0

SAMPLE_HZ = 200            # integration rate; steady & high to limit drift
CALIB_SAMPLES = 2000       # ~10 s of stillness at 200 Hz for a solid bias


def _twos_complement(high, low):
    value = (high << 8) | low
    return value - 0x10000 if value >= 0x8000 else value


class YawTracker:
    """
    Owns the MPU6050, calibrates gx bias, and integrates yaw in a background
    thread. Thread-safe read via get_yaw().
    """

    def __init__(self, bus):
        self.bus = bus
        self.gx_bias = 0.0           # deg/s, removed from every sample
        self._yaw = 0.0              # radians, integrated heading (yaw_seg3)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._init_mpu()

    # --- hardware setup (same as Step 1) ---
    def _init_mpu(self):
        self.bus.write_byte_data(MPU_ADDR, REG_PWR_MGMT_1, 0x01)  # wake, PLL clk
        time.sleep(0.1)
        self.bus.write_byte_data(MPU_ADDR, REG_SMPLRT_DIV, 0x00)
        self.bus.write_byte_data(MPU_ADDR, REG_CONFIG, 0x03)       # DLPF ~42 Hz
        self.bus.write_byte_data(MPU_ADDR, REG_GYRO_CONFIG, 0x00)  # +/-250 dps
        self.bus.write_byte_data(MPU_ADDR, REG_ACCEL_CONFIG, 0x00)
        time.sleep(0.05)

    def _read_gx_dps(self):
        """Read only the gx word and return deg/s (raw, bias NOT removed)."""
        raw = self.bus.read_i2c_block_data(MPU_ADDR, REG_ACCEL_XOUT_H, 14)
        gx = _twos_complement(raw[8], raw[9])    # gyro X is bytes [8],[9]
        return gx / GYRO_SENS

    # --- step 2a: bias calibration ---
    def calibrate(self, n=CALIB_SAMPLES):
        """
        Average gx with the robot DEAD STILL. The resting gx (~ -4.9 dps in
        your test) is the bias we subtract from every future sample. Skipping
        this walks the heading several deg/s.
        """
        print(f"[CALIB] Hold the robot COMPLETELY STILL. "
              f"Averaging {n} samples (~{n/SAMPLE_HZ:.0f}s)...")
        period = 1.0 / SAMPLE_HZ
        acc = 0.0
        # also track spread to warn if the robot moved during calibration
        vmin, vmax = float("inf"), float("-inf")
        for _ in range(n):
            v = self._read_gx_dps()
            acc += v
            vmin, vmax = min(vmin, v), max(vmax, v)
            time.sleep(period)
        self.gx_bias = acc / n
        spread = vmax - vmin
        print(f"[CALIB] gx bias = {self.gx_bias:+.3f} dps "
              f"(sample spread {spread:.2f} dps)")
        if spread > 2.0:
            print("[CALIB][WARN] Large spread - was it really still? "
                  "Re-run calibration.")
        return self.gx_bias

    # --- step 2b: integration thread ---
    def _loop(self):
        period = 1.0 / SAMPLE_HZ
        last = time.monotonic()
        while self._running:
            now = time.monotonic()
            dt = now - last          # REAL elapsed time, not assumed 1/HZ
            last = now

            rate_dps = (self._read_gx_dps() - self.gx_bias) * YAW_SIGN
            dyaw = math.radians(rate_dps) * dt

            with self._lock:
                self._yaw = self._normalize(self._yaw + dyaw)

            # sleep the remainder of the period (best-effort steady rate)
            sleep_left = period - (time.monotonic() - now)
            if sleep_left > 0:
                time.sleep(sleep_left)

    @staticmethod
    def _normalize(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def get_yaw(self):
        """Thread-safe current heading (radians), yaw_seg3."""
        with self._lock:
            return self._yaw

    def get_yaw_deg(self):
        return math.degrees(self.get_yaw())

    def reset_yaw(self, value_rad=0.0):
        """Used later by checkpoint snapping to overwrite heading."""
        with self._lock:
            self._yaw = self._normalize(value_rad)


def main():
    with SMBus(I2C_BUS) as bus:
        tracker = YawTracker(bus)
        tracker.calibrate()
        tracker.reset_yaw(0.0)      # define "now" as 0 deg for the test
        tracker.start()

        print("\n[RUN] Integrating yaw on gx. Ctrl+C to stop.")
        print("      TEST 1 (protractor): rotate a known angle, check readback.")
        print("      TEST 2 (drift): leave still, watch deg/min accumulate.\n")
        try:
            while True:
                print(f"   yaw_seg3 = {tracker.get_yaw_deg():+8.2f} deg", end="\r")
                time.sleep(0.05)
        except KeyboardInterrupt:
            tracker.stop()
            print(f"\n[INFO] Final yaw = {tracker.get_yaw_deg():+.2f} deg. Stopped.")


if __name__ == "__main__":
    main()
