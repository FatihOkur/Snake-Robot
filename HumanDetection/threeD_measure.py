import cv2
import cv2.aruco as aruco
import numpy as np
import json
import subprocess
import socket
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
MARKER_SIZE = 0.21  # 3.5 cm
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
aruco_params = aruco.DetectorParameters()

# --- CHECKPOINT UDP BROADCAST (for StateEstimator in the UART feeder) ---
CHECKPOINT_UDP_IP   = "127.0.0.1"
CHECKPOINT_UDP_PORT = 5005
_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- CAMERA-TO-MAP TRANSFORM (PLACEHOLDER — calibrate on robot!) ---
# These offsets and scale convert the camera-frame (x_m, z_m) into the
# planner's map coordinates.  Adjust after mounting and measuring.
MAP_ORIGIN_X = 17.0      # map x of the camera's optical center
MAP_ORIGIN_Y = 10.0      # map y of the camera's optical center
MAP_SCALE    = 10.0       # metres -> map units (1 m = 10 dm, 1 PU = 10 cm)
CAMERA_YAW_OFFSET = 0.0   # radians: rotation between camera frame and map frame

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

                # --- CHECKPOINT BROADCAST (Task 4) ---
                # Convert camera-frame pose to map coordinates.
                # PLACEHOLDER TRANSFORM — adjust after physical calibration.
                map_x = MAP_ORIGIN_X + x_m * MAP_SCALE
                map_y = MAP_ORIGIN_Y + z_m * MAP_SCALE
                # Derive yaw from rotation vector
                rmat, _ = cv2.Rodrigues(rvecs[i])
                map_yaw = math.atan2(rmat[1, 0], rmat[0, 0]) + CAMERA_YAW_OFFSET

                checkpoint_msg = json.dumps({
                    "x": round(float(map_x), 4),
                    "y": round(float(map_y), 4),
                    "yaw_rad": round(float(map_yaw), 6),
                })
                try:
                    _udp_sock.sendto(
                        checkpoint_msg.encode("utf-8"),
                        (CHECKPOINT_UDP_IP, CHECKPOINT_UDP_PORT),
                    )
                except Exception:
                    pass  # non-critical; feeder may not be running

        cv2.imshow("Yilan Robot - Checkpoint Tespiti", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Kapanış işlemleri
process.terminate()
cv2.destroyAllWindows()