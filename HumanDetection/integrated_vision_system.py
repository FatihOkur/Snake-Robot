import cv2
import cv2.aruco as aruco
import numpy as np
import json
import subprocess
import socket
import math
from ultralytics import YOLO

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================

print("[SYSTEM] Booting Integrated Vision System (YOLO + ArUco)...")

# --- YOLO Setup ---
model = YOLO('yolo11n-pose.pt')
PROCESS_EVERY_N_FRAMES = 3 

# --- ArUco Calibration Setup ---
try:
    with open('camera_calibration.json', 'r') as f:
        calib_data = json.load(f)
        camera_matrix = np.array(calib_data["camera_matrix"])
        dist_coeffs = np.array(calib_data["dist_coeff"])
        print("[INFO] Camera calibration loaded successfully.")
except:
    print("[ERROR] camera_calibration.json not found! Exiting.")
    exit()

MARKER_SIZE = 0.21  # 3.5 cm
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
aruco_params = aruco.DetectorParameters()

# --- Checkpoint Dictionary ---
CHECKPOINT_LOCATIONS = {
    0: {"x": 17.0, "y": 20.0, "name": "Start/Entry"},
    1: {"x": 25.0, "y": 35.0, "name": "Intersection"},
    2: {"x": 20.0, "y": 50.0, "name": "Debris Climb Zone"},
    3: {"x": 26.0, "y": 60.0, "name": "Goal/Docking Station"}
}

# --- UDP / Map Setup ---
CHECKPOINT_UDP_IP   = "127.0.0.1"
CHECKPOINT_UDP_PORT = 5005
_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
MAP_SCALE    = 10.0       
CAMERA_YAW_OFFSET = 0.0   

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def enhance_frame(frame):
    if frame is None: return None
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# ==========================================
# 3. MAIN CAMERA LOOP
# ==========================================
print("[VISION] Starting hardware camera bypass (rpicam-vid)...")
cmd = [
    'rpicam-vid', '-t', '0', 
    '--codec', 'mjpeg', 
    '--width', '640', 
    '--height', '480', 
    '--framerate', '30', 
    '--vflip', '--hflip', 
    '-o', '-'
]

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
bytes_data = b''
frame_count = 0
display_frame = None

print("[SYSTEM] Unified Radar Active. Press 'q' to quit.")

while True:
    # Read image bytes
    bytes_data += process.stdout.read(4096)
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')
    
    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:] 
        
        # Decode Frame
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None: continue
        
        frame_count += 1
        enhanced_frame = enhance_frame(frame)
        
        # --------------------------------------------------
        # TASK A: YOLO HUMAN DETECTION (Every N Frames)
        # --------------------------------------------------
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results = model.predict(enhanced_frame, conf=0.25, imgsz=320, verbose=False)
            display_frame = results[0].plot() # This draws YOLO boxes on the frame
        elif display_frame is None:
            display_frame = enhanced_frame.copy()
            
        # If we skipped YOLO this frame, we still want to use the fresh frame as our base for ArUco
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            display_frame = enhanced_frame.copy()

        # --------------------------------------------------
        # TASK B: ARUCO CHECKPOINT LOCALIZATION (Every Frame)
        # --------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                # Draw axes and boxes directly onto the display_frame (which already has YOLO boxes)
                aruco.drawDetectedMarkers(display_frame, corners)
                cv2.drawFrameAxes(display_frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                x_m, y_m, z_m = tvecs[i][0][0], tvecs[i][0][1], tvecs[i][0][2]
                marker_id = ids[i][0]

                # Render Text
                cv2.putText(display_frame, f"ID: {marker_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Z: {z_m*100:.1f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Checkpoint Broadcast
                if marker_id in CHECKPOINT_LOCATIONS:
                    loc = CHECKPOINT_LOCATIONS[marker_id]
                    
                    robot_absolute_x = loc['x'] - (x_m * MAP_SCALE) 
                    robot_absolute_y = loc['y'] - (z_m * MAP_SCALE) 
                    
                    rmat, _ = cv2.Rodrigues(rvecs[i])
                    map_yaw = math.atan2(rmat[1, 0], rmat[0, 0]) + CAMERA_YAW_OFFSET

                    checkpoint_msg = json.dumps({
                        "id": int(marker_id),
                        "name": loc['name'],
                        "x": round(float(robot_absolute_x), 4),
                        "y": round(float(robot_absolute_y), 4),
                        "yaw_rad": round(float(map_yaw), 6)
                    })
                    
                    try:
                        _udp_sock.sendto(checkpoint_msg.encode("utf-8"), (CHECKPOINT_UDP_IP, CHECKPOINT_UDP_PORT))
                    except Exception:
                        pass

        # --------------------------------------------------
        # RENDER UNIFIED WINDOW
        # --------------------------------------------------
        cv2.imshow('Snake Robot - Unified Vision (YOLO + SLAM)', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[SYSTEM] User requested shutdown...")
            break

process.terminate()
cv2.destroyAllWindows()
print("[SYSTEM] Live feed closed. Ready for next command.")