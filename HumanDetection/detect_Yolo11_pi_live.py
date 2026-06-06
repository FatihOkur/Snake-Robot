import cv2
import numpy as np
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

    model = YOLO("yolo11n-pose.pt")
    PROCESS_EVERY_N_FRAMES = 3

    print("[VISION] Starting live camera feed...")

    # libcamerify KULLANMA — düz V4L2 + YUYV
    # YUYV, libcamera'nın PiSP backend'inin desteklediği native format
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("[ERROR] Kamera açılamadı! /dev/video0 mevcut mu kontrol et.")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Gerçekte ne ayarlandığını logla
    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[VISION] Stream: {actual_w}x{actual_h} @ {actual_fps:.1f} fps")

    frame_count   = 0
    display_frame = None

    while True:
        ret, frame = cap.read()

        if not ret or frame is None or frame.size == 0:
            print("[WARNING] Boş kare, atlanıyor...")
            continue

        # Kamera fiziksel olarak 90° yan duruyorsa bu satırı aç:
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_count += 1
        enhanced_frame = enhance_frame(frame)

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results       = model.predict(enhanced_frame, conf=0.25, imgsz=320, verbose=False)
            display_frame = results[0].plot()
        else:
            display_frame = enhanced_frame

        cv2.imshow("YOLO11 Pose Estimation - Live", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Live feed kapatıldı.")


if __name__ == "__main__":
    main()