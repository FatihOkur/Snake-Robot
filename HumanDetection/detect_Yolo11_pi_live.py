import cv2
from ultralytics import YOLO
import numpy as np

def enhance_frame(frame):
    if frame is None:
        return None
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def main():
    print("[SYSTEM] Booting EDGE-OPTIMIZED YOLO11 Pose Estimation for ARM CPU...")
    model = YOLO('yolo11n-pose.pt')
    
    print("[VISION] Initializing camera interface...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # KRİTİK EKLENTİ: Kameradan veriyi MJPEG olarak talep et (Boş kare hatasını önler)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    # Çözünürlüğü sabitle (Reshape hatasını önler)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Ensure you are running with 'libcamerify'")
        return

    PROCESS_EVERY_N_FRAMES = 3 
    frame_count = 0
    display_frame = None

    print("[SYSTEM] Scan complete. Live feed starting. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        
        # Boş frame kontrolü
        if not ret or frame is None or frame.size == 0:
            print("[WARN] Received empty frame, skipping...")
            continue
            
        frame_count += 1
        
        enhanced_frame = enhance_frame(frame)
        
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results = model.predict(enhanced_frame, conf=0.25, imgsz=320, verbose=False)
            display_frame = results[0].plot()
        elif display_frame is None:
            display_frame = enhanced_frame
            
        cv2.imshow('Snake Robot - Live Human Detection', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[SYSTEM] User requested shutdown...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Live feed closed. Ready for next command.")

if __name__ == "__main__":
    main()