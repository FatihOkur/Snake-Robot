import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO


# ─────────────────────────────────────────────
#  Image enhancement (CLAHE on L-channel)
# ─────────────────────────────────────────────
def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement in LAB colour space."""
    if frame is None:
        return None
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    print("[SYSTEM] Booting EDGE-OPTIMIZED YOLO11 Pose Estimation for ARM CPU...")

    # ── Model ──────────────────────────────────
    # yolo11n-pose.pt  →  nano variant, lowest memory & CPU footprint
    model = YOLO("yolo11n-pose.pt")

    # Run inference only on every Nth frame to reduce CPU load
    PROCESS_EVERY_N_FRAMES = 3

    # ── Camera (picamera2 — no libcamerify needed) ──
    print("[VISION] Starting live camera feed via picamera2...")

    picam2 = Picamera2()

    # BGR888 gives us a standard OpenCV-compatible 3-channel array directly.
    # No MJPG / YUYV negotiation → no PiSP pixel-format crash.
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()
    print("[VISION] Camera started successfully.")

    frame_count   = 0
    display_frame = None          # keeps the last annotated frame during skipped frames

    while True:
        # capture_array() returns a numpy ndarray — no VideoCapture required
        frame = picam2.capture_array()

        if frame is None or frame.size == 0:
            print("[WARNING] Empty frame received, skipping...")
            continue

        # Uncomment the line below if the camera is physically mounted 90° sideways
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_count += 1
        enhanced_frame = enhance_frame(frame)

        # ── Inference (every Nth frame only) ───
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # imgsz=320 keeps inference fast on ARM; raise to 480 for more accuracy
            results       = model.predict(enhanced_frame, conf=0.25, imgsz=320, verbose=False)
            display_frame = results[0].plot()
        else:
            # Show the enhanced frame while skipping inference
            display_frame = enhanced_frame

        cv2.imshow("YOLO11 Pose Estimation - Live", display_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ────────────────────────────────
    picam2.stop()
    cv2.destroyAllWindows()
    print("[SYSTEM] Live feed closed.")


if __name__ == "__main__":
    main()