import cv2
from ultralytics import YOLO

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

    # 1. DOWNGRADE TO NANO MODEL for SBC compatibility
    model = YOLO('yolo11n-pose.pt')
    
    PROCESS_EVERY_N_FRAMES = 3 

    print("[VISION] Starting live camera feed...")
    
    # GStreamer iptal, doğrudan standart portu kullanıyoruz
    cap = cv2.VideoCapture(0)
    
    # Çözünürlüğü manuel olarak düşürüyoruz ki FPS yüksek kalsın
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("[WARNING] Empty frame received, skipping...")
            continue
        
        # Eğer kamera fiziksel olarak 90 derece yan duruyorsa bu satırı aktif et
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_count += 1
        enhanced_frame = enhance_frame(frame)
        
        # Sadece her N. karede yapay zekayı çalıştır
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results = model.predict(enhanced_frame, conf=0.25, imgsz=320, verbose=False)
            display_frame = results[0].plot()
        else:
            display_frame = enhanced_frame
        
        cv2.imshow("YOLO11 Pose Estimation - Live", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Live feed closed.")

if __name__ == "__main__":
    main()