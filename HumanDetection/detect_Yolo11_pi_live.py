import cv2
import numpy as np
import sys
from ultralytics import YOLO


# ─────────────────────────────────────────────
#  Image enhancement (CLAHE on L-channel)
# ─────────────────────────────────────────────
def enhance_frame(frame: np.ndarray) -> np.ndarray:
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

    # Pi 5 rp1-cfe kamerası libcamera pipeline üzerinden çalışır.
    # stdin'den gelen MJPEG stream'i okuyoruz (libcamera-vid pipe ile).
    # Eğer stdin'den veri geliyorsa pipe modunu kullan, yoksa fallback dene.
    pipe_mode = not sys.stdin.isatty()

    if pipe_mode:
        print("[VISION] Pipe modu: libcamera-vid stdin'den okunuyor...")
        cap = cv2.VideoCapture("/dev/stdin")
    else:
        # Fallback: direkt device dene (başka bir sistemde işe yarayabilir)
        print("[VISION] Direkt mod: /dev/video0 deneniyor...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Kamera açılamadı!")
        print("[HINT] Şu komutla çalıştır:")
        print("  libcamera-vid -t 0 --width 640 --height 480 --codec mjpeg --inline -o - | python detect_Yolo11_pi_live.py")
        return

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[VISION] Stream: {actual_w}x{actual_h} @ {actual_fps:.1f} fps")

    frame_count   = 0
    display_frame = None
    empty_count   = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None or frame.size == 0:
            empty_count += 1
            if empty_count > 30:
                print("[ERROR] 30 ardışık boş kare — kamera bağlantısı kesildi, çıkılıyor.")
                break
            continue

        empty_count = 0  # başarılı kare geldi, sayacı sıfırla

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