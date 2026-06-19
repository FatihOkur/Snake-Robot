import cv2
import cv2.aruco as aruco
import numpy as np
import json
import subprocess

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
MARKER_SIZE = 0.21  # 3.5 cm
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
aruco_params = aruco.DetectorParameters()

# --- 3. DONANIM SEVİYESİNDE KAMERAYI BAŞLAT (RPICAM-VID) ---
print("[VİZYON] Donanım kamera atlaması (rpicam-vid) başlatılıyor...")
cmd = [
    'rpicam-vid', '-t', '0', 
    '--codec', 'mjpeg', 
    '--width', '640', 
    '--height', '480', 
    '--framerate', '30', 
    '--vflip',        # Görüntüyü dikey çevir (Donanımsal)
    '--hflip',        # Görüntüyü yatay çevir (Donanımsal)
    '-o', '-'
]

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
bytes_data = b''

print("[SİSTEM] Checkpoint Radar aktif. Çıkmak için 'q' tuşuna basın.")

while True:
    # Görüntü baytlarını oku
    bytes_data += process.stdout.read(4096)
    
    # JPEG çerçevesinin başlangıç ve bitişini bul
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')
    
    if a != -1 and b != -1:
        # Tam kare (frame) yakalandı
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:] 
        
        # JPEG'i OpenCV formatına çevir
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if frame is None:
            continue

        # Görüntüyü gri tonlamaya çevir (ArUco için)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Marker'ları ara
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            # Etiket bulunduysa 3D pozisyonunu hesapla
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                # Çerçeve ve eksenleri çiz
                aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                # Mesafeleri hesapla
                x_m = tvecs[i][0][0]
                y_m = tvecs[i][0][1]
                z_m = tvecs[i][0][2]

                distance_str = f"Mesafe (Z): {z_m*100:.1f} cm"
                offset_str   = f"Yatay Kayma (X): {x_m*100:.1f} cm"

                cv2.putText(frame, f"ID: {ids[i][0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, distance_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, offset_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Yilan Robot - Checkpoint Tespiti", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Kapanış işlemleri
process.terminate()
cv2.destroyAllWindows()