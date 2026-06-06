import cv2
from ultralytics import YOLO
import numpy as np
import subprocess

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
    
    print("[VISION] Starting hardware camera bypass (rpicam-vid)...")
    
    # Raspberry Pi'nin KENDİ kamera programını arka planda başlatıyoruz
    # Görüntüyü MJPEG formatında standart çıktıya (stdout) yönlendiriyoruz
    cmd = [
        'rpicam-vid', '-t', '0', 
        '--codec', 'mjpeg', 
        '--width', '640', 
        '--height', '480', 
        '--framerate', '30', 
        '--vflip',        # Görüntüyü dikey çevir
        '--hflip',        # Görüntüyü yatay çevir
        '-o', '-'
    ]
    
    # Kamerayı çalıştır
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    PROCESS_EVERY_N_FRAMES = 3 
    frame_count = 0
    display_frame = None
    bytes_data = b''

    print("[SYSTEM] Scan complete. Live feed starting. Press 'q' to quit.")

    while True:
        # Görüntü baytlarını oku
        bytes_data += process.stdout.read(4096)
        
        # Bir JPEG dosyasının başlangıç (\xff\xd8) ve bitiş (\xff\xd9) noktalarını bul
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        
        if a != -1 and b != -1:
            # Tam bir kare (frame) yakalandı
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:] # Okunan kısmı tampondan sil
            
            # JPEG'i OpenCV formatına çevir
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            frame_count += 1
            enhanced_frame = enhance_frame(frame)
            
            # YOLO Algoritmasını çalıştır
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                results = model.predict(enhanced_frame, conf=0.25, imgsz=320, verbose=False)
                display_frame = results[0].plot()
            elif display_frame is None:
                display_frame = enhanced_frame
                
            # Görüntüyü ekrana yansıt
            cv2.imshow('Snake Robot - Live Human Detection', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[SYSTEM] User requested shutdown...")
                break

    # Kapanış işlemleri
    process.terminate()
    cv2.destroyAllWindows()
    print("[SYSTEM] Live feed closed. Ready for next command.")

if __name__ == "__main__":
    main()