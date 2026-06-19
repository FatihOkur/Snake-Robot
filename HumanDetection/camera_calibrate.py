import numpy as np
import cv2
import glob
import json
import os

# --- AYARLAR ---
# Satranç tahtasındaki İÇ KÖŞE sayısı. (Dış kenarlar sayılmaz!)
CHECKERBOARD = (8, 6) 

# Tek bir karenin GERÇEK DÜNYA boyutu (Metre cinsinden. 29mm = 0.029m)
SQUARE_SIZE = 0.029  

# Fotoğrafların bulunduğu klasör
IMAGE_DIR = 'calibration_images/*.jpg'
# ---------------

# 3D gerçek dünya noktalarını hazırlıyoruz
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Tüm resimlerdeki 3D gerçek dünya noktalarını ve 2D görüntü (piksel) noktalarını saklayacak diziler
objpoints = [] # Gerçek dünyadaki 3D noktalar
imgpoints = [] # Görüntü düzlemindeki 2D noktalar (Piksel)

# Alt-piksel hassasiyeti için kriterler
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

images = glob.glob(IMAGE_DIR)

if len(images) == 0:
    print("[HATA] Belirtilen klasörde hiç fotoğraf bulunamadı!")
    exit()

print(f"[BİLGİ] Toplam {len(images)} fotoğraf işleniyor...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Satranç tahtası köşelerini bul
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Kose Tespiti (Gecmek icin tusa basin)', img)
        cv2.waitKey(100) 
        
cv2.destroyAllWindows()

# Yeterli veri toplandıysa kamerayı kalibre et
if len(objpoints) > 0:
    print("[BİLGİ] Kalibrasyon hesaplanıyor. Bu işlem Pi üzerinde birkaç saniye sürebilir...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("\n--- KALİBRASYON SONUÇLARI ---")
    print("RMS Hata Oranı:", ret)
    print("\nKamera Matrisi (Intrinsic Parameters):\n", mtx)
    print("\nHesaplanan (Ancak Yoksayılacak) Bozulma Katsayıları:\n", dist)
    
    # --- 8mm LENS İÇİN ÖZEL MÜDAHALE ---
    print("\n[BİLGİ] Dar Açılı 8mm (42°) Lens kullanıldığı için Bozulma Katsayıları otomatik olarak SIFIRLANIYOR...")
    
    # Parametreleri JSON formatında kaydet
    calibration_data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeff": [[0.0, 0.0, 0.0, 0.0, 0.0]]  # <- BURAYI MANUEL SIFIRA SABİTLEDİK
    }
    
    with open("camera_calibration.json", "w") as f:
        json.dump(calibration_data, f, indent=4)
        
    print("\n[BAŞARILI] Kalibrasyon verileri (Bozulma 0.0 olarak) 'camera_calibration.json' dosyasına kaydedildi.")
else:
    print("[HATA] Hiçbir resimde satranç tahtası tam olarak algılanamadı.")