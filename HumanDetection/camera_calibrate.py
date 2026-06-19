import numpy as np
import cv2
import glob
import json
import os

# --- AYARLAR ---
# Satranç tahtasındaki İÇ KÖŞE sayısı. (Dış kenarlar sayılmaz!)
# PDF'inize göre 9x7 kare var, bu da 8x6 iç köşe yapar.
CHECKERBOARD = (8, 6) 

# Tek bir karenin GERÇEK DÜNYA boyutu (Metre cinsinden. 30mm = 0.030m)
SQUARE_SIZE = 0.029  

# Fotoğrafların bulunduğu klasör
IMAGE_DIR = 'calibration_images/*jpg.jpeg'
# ---------------

# 3D gerçek dünya noktalarını hazırlıyoruz: (0,0,0), (1,0,0), (2,0,0) ...
# Daha sonra bunları SQUARE_SIZE ile çarparak gerçek metrik boyutlara getireceğiz.
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Tüm resimlerdeki 3D gerçek dünya noktalarını ve 2D görüntü (piksel) noktalarını saklayacak diziler
objpoints = [] # Gerçek dünyadaki 3D noktalar
imgpoints = [] # Görüntü düzlemindeki 2D noktalar (Piksel)

# Alt-piksel hassasiyeti için kriterler (Optimizasyon durma koşulları)
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
    
    # Eğer köşeler bulunursa, objpoints ve imgpoints listelerine ekle
    if ret == True:
        objpoints.append(objp)
        
        # Köşe koordinatlarını alt-piksel seviyesinde hassaslaştır
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Bulunan köşeleri ekranda çiz ve göster (Görsel doğrulama için)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Kose Tespiti (Gecmek icin tusa basin)', img)
        cv2.waitKey(100) # Her resmi 100ms göster
        
cv2.destroyAllWindows()

# Yeterli veri toplandıysa kamerayı kalibre et
if len(objpoints) > 0:
    print("[BİLGİ] Kalibrasyon hesaplanıyor. Bu işlem Pi üzerinde birkaç saniye sürebilir...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("\n--- KALİBRASYON SONUÇLARI ---")
    print("RMS Hata Oranı:", ret) # 1.0'ın altındaysa kalibrasyon çok iyidir.
    print("\nKamera Matrisi (Intrinsic Parameters):\n", mtx)
    print("\nBozulma Katsayıları (Distortion Coefficients):\n", dist)
    
    # Parametreleri JSON formatında kaydet
    calibration_data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeff": dist.tolist()
    }
    
    with open("camera_calibration.json", "w") as f:
        json.dump(calibration_data, f, indent=4)
        
    print("\n[BAŞARILI] Kalibrasyon verileri 'camera_calibration.json' dosyasına kaydedildi.")
else:
    print("[HATA] Hiçbir resimde satranç tahtası tam olarak algılanamadı.")