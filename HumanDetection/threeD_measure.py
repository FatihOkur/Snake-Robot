import cv2
import cv2.aruco as aruco
import numpy as np
import json
import math

# --- 1. KALİBRASYON VERİLERİNİ YÜKLE ---
try:
    with open('camera_calibration.json', 'r') as f:
        calib_data = json.load(f)
        camera_matrix = np.array(calib_data["camera_matrix"])
        dist_coeffs = np.array(calib_data["dist_coeff"])
        print("[BİLGİ] Kalibrasyon verileri başarıyla yüklendi.")
except:
    print("[HATA] camera_calibration.json bulunamadı! Lütfen önce kalibrasyon yapın.")
    exit()

# --- 2. CHECKPOINT (ARUCO) AYARLARI ---
# Siyah karenin dış kenar uzunluğu metre cinsinden (3.5 cm = 0.035 m)
MARKER_SIZE = 0.035  

# ArUco sözlüğünü tanımla (Bastırdığınız PDF'ler DICT_4X4_1000 ailesinden)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
aruco_params = aruco.DetectorParameters()

# --- 3. KAMERAYI BAŞLAT ---
cap = cv2.VideoCapture(0)
# Kalibrasyon fotoğraflarını çektiğiniz çözünürlük (Örn: 640x480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[SİSTEM] Checkpoint Radar aktif. Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Marker'ları ara
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        # Etiket bulunduysa, pozisyonunu tahmin et (Pose Estimation)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            # Tespiti ekranda yeşil çerçeve ile çiz ve 3D eksenleri (X,Y,Z) göster
            aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

            # tvecs[i][0] bize kameraya olan mesafeleri [X, Y, Z] olarak (Metre cinsinden) verir.
            # Z: Kameradan ileri olan UZAKLIK
            # X: Kameranın sağı/solu yatay kayması
            x_m = tvecs[i][0][0]
            y_m = tvecs[i][0][1]
            z_m = tvecs[i][0][2]

            # Ekrana yazdırılacak metinleri hazırla (Santimetreye çevirerek)
            distance_str = f"Mesafe (Z): {z_m*100:.1f} cm"
            offset_str   = f"Yatay Kayma (X): {x_m*100:.1f} cm"

            cv2.putText(frame, f"ID: {ids[i][0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, distance_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, offset_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Yilan Robot - Checkpoint Tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()