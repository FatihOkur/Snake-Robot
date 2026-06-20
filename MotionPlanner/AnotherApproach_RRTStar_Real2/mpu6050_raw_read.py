#!/usr/bin/env python3
"""
Step 1: Raw MPU6050 read over I2C on the Raspberry Pi.

Goal of THIS script (deliberately minimal):
  - Confirm the sensor is on the bus and talking.
  - Wake it from sleep and read the 6 raw values (3 accel, 3 gyro) + temp.
  - Print scaled, human-readable numbers at a steady rate.

NO bias calibration, NO yaw integration here. That is Step 2.
We only want to prove the data path and sanity-check axis behaviour.

Mounting reminder: IMU is on the SEGMENT 3 face of joint 3, so the gyro
axis normal to the floor reads yaw_seg3 directly. With a flat mount, that
is the Z axis. Note which physical axis points "up" when you run this.
"""

import time
from smbus2 import SMBus

# ---------------------------------------------------------------------------
# I2C / MPU6050 constants
# ---------------------------------------------------------------------------
I2C_BUS = 1                # Raspberry Pi default user I2C bus
MPU_ADDR = 0x68            # AD0 low = 0x68, AD0 high = 0x69

# Register map (only what we need)
REG_PWR_MGMT_1   = 0x6B    # power management / sleep / clock source
REG_WHO_AM_I     = 0x75    # identity register, should read 0x68
REG_SMPLRT_DIV   = 0x19    # sample rate divider
REG_CONFIG       = 0x1A    # DLPF config
REG_GYRO_CONFIG  = 0x1B    # gyro full-scale select
REG_ACCEL_CONFIG = 0x1C    # accel full-scale select
REG_ACCEL_XOUT_H = 0x3B    # first byte of the 14-byte burst (accel/temp/gyro)

# Full-scale selections we will program.
# Gyro  +/-250 dps  -> sensitivity 131   LSB/(deg/s)
# Accel +/-2 g      -> sensitivity 16384 LSB/g
GYRO_FS_SEL   = 0x00       # 0 = +/-250 dps
ACCEL_FS_SEL  = 0x00       # 0 = +/-2 g
GYRO_SENS  = 131.0         # LSB per deg/s
ACCEL_SENS = 16384.0       # LSB per g

READ_HZ = 50               # print rate for this smoke test


def _twos_complement(high, low):
    """Combine two bytes into a signed 16-bit value."""
    value = (high << 8) | low
    if value >= 0x8000:
        value -= 0x10000
    return value


def init_mpu(bus, addr=MPU_ADDR):
    """Wake the device and set sane full-scale ranges + a light DLPF."""
    # Wake up: clear SLEEP bit, select gyro-X PLL as clock (more stable than
    # the internal 8 MHz oscillator).
    bus.write_byte_data(addr, REG_PWR_MGMT_1, 0x01)
    time.sleep(0.1)

    # Sample-rate divider: with DLPF enabled the gyro output rate is 1 kHz,
    # divider 0 -> 1 kHz internal. We downsample in software.
    bus.write_byte_data(addr, REG_SMPLRT_DIV, 0x00)
    # DLPF ~44 Hz accel / 42 Hz gyro: trims high-frequency motor/track noise.
    bus.write_byte_data(addr, REG_CONFIG, 0x03)
    bus.write_byte_data(addr, REG_GYRO_CONFIG, GYRO_FS_SEL)
    bus.write_byte_data(addr, REG_ACCEL_CONFIG, ACCEL_FS_SEL)
    time.sleep(0.05)


def who_am_i(bus, addr=MPU_ADDR):
    return bus.read_byte_data(addr, REG_WHO_AM_I)


def read_block(bus, addr=MPU_ADDR):
    """
    Burst-read 14 bytes starting at ACCEL_XOUT_H:
      [0:6]   accel X/Y/Z (H,L each)
      [6:8]   temperature
      [8:14]  gyro  X/Y/Z (H,L each)
    Burst read guarantees all axes come from the SAME sample.
    """
    raw = bus.read_i2c_block_data(addr, REG_ACCEL_XOUT_H, 14)

    ax = _twos_complement(raw[0],  raw[1])
    ay = _twos_complement(raw[2],  raw[3])
    az = _twos_complement(raw[4],  raw[5])
    temp_raw = _twos_complement(raw[6], raw[7])
    gx = _twos_complement(raw[8],  raw[9])
    gy = _twos_complement(raw[10], raw[11])
    gz = _twos_complement(raw[12], raw[13])

    accel_g = (ax / ACCEL_SENS, ay / ACCEL_SENS, az / ACCEL_SENS)
    gyro_dps = (gx / GYRO_SENS, gy / GYRO_SENS, gz / GYRO_SENS)
    temp_c = temp_raw / 340.0 + 36.53      # datasheet formula
    return accel_g, gyro_dps, temp_c


def main():
    with SMBus(I2C_BUS) as bus:
        ident = who_am_i(bus)
        print(f"[INFO] WHO_AM_I = 0x{ident:02X} (expect 0x68)")
        if ident != 0x68:
            print("[WARN] Unexpected WHO_AM_I. Check address (AD0 pin) and wiring.")

        init_mpu(bus)
        print("[INFO] MPU6050 awake. Reading raw data. Ctrl+C to stop.\n")
        print(f"{'ax(g)':>8}{'ay(g)':>8}{'az(g)':>8}"
              f"{'gx(/s)':>9}{'gy(/s)':>9}{'gz(/s)':>9}{'T(C)':>7}")

        period = 1.0 / READ_HZ
        while True:
            accel_g, gyro_dps, temp_c = read_block(bus)
            print(f"{accel_g[0]:8.3f}{accel_g[1]:8.3f}{accel_g[2]:8.3f}"
                  f"{gyro_dps[0]:9.2f}{gyro_dps[1]:9.2f}{gyro_dps[2]:9.2f}"
                  f"{temp_c:7.1f}", end="\r")
            time.sleep(period)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")