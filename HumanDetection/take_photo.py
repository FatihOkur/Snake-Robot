import cv2
import numpy as np
import subprocess
import os

# --- 1. KAYIT KLASÖRÜNÜ OLUŞTUR ---
SAVE_DIR = "calibration_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"[BİLGİ] '{SAVE_DIR}' klasörü oluşturuldu.")

# --- 2. KAMERAYI BAŞLAT (RPICAM-VID) ---
print("[VİZYON] Kamera kalibrasyon çekimi için başlatılıyor...")
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

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
bytes_data = b''
saved_count = 0

print("\n--- KULLANIM REHBERİ ---")
print("Fotoğraf çekmek için: 's' tuşuna basın.")
print("Çıkış yapmak için: 'q' tuşuna basın.")
print("Hedef: Farklı açılardan 15-20 fotoğraf çekmek.\n")

while True:
    # Görüntü baytlarını oku
    bytes_data += process.stdout.read(4096)
    
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')
    
    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:] 
        
        # JPEG'i OpenCV formatına çevir
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if frame is None:
            continue

        # Ekrana kullanıcı için bilgi metinleri ekleyelim (Sadece ekranda görünür, fotoğrafa kaydedilmez)
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Cekilen Fotograf: {saved_count}/20", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, "Cekmek icin 's', Cikmak icin 'q'", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Kalibrasyon Cekimi", display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # 's' tuşuna basılırsa TEMİZ (yazısız) fotoğrafı kaydet
        if key == ord('s'):
            filepath = os.path.join(SAVE_DIR, f"calib_img_{saved_count:02d}.jpg")
            cv2.imwrite(filepath, frame)
            saved_count += 1
            print(f"[KAYDEDİLDİ] {filepath} ({saved_count}. fotoğraf)")
            
        # 'q' tuşuna basılırsa çık
        elif key == ord('q'):
            print(f"\n[SİSTEM] Çıkış yapılıyor. Toplam {saved_count} fotoğraf çekildi.")
            break

# Kapanış işlemleri
process.terminate()
cv2.destroyAllWindows()