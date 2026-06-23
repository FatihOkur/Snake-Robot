import cv2
import cv2.aruco as aruco
import numpy as np
import json
import subprocess
import socket
import math
import threading
import time
from ultralytics import YOLO

# ==========================================
# 1. THREADED CAMERA STREAM (THE LAG KILLER)
# ==========================================
class FastPiCamStream:
    def __init__(self):
        print("[VISION] Starting hardware camera bypass (Threaded)...")
        cmd = [
            'rpicam-vid', '-t', '0', 
            '--codec', 'mjpeg', 
            '--width', '640', 
            '--height', '480', 
            '--framerate', '30', 
            '-o', '-'
        ]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.latest_frame = None
        self.running = True
        
        # Start background thread to clear the buffer
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        bytes_data = b''
        while self.running:
            bytes_data += self.process.stdout.read(4096)
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:] 
                
                # Decode and ALWAYS update the latest frame, dropping intermediate ones
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    self.latest_frame = frame

    def stop(self):
        self.running = False
        self.process.terminate()

# ==========================================
# 2. INITIALIZATION & CONFIGURATION
# ==========================================
print("[SYSTEM] Booting Optimized Vision System (YOLO + ArUco)...")

model = YOLO('yolo11n-pose.pt')
PROCESS_EVERY_N_FRAMES = 5  # Increased to give the Pi more breathing room

try:
    with open('camera_calibration.json', 'r') as f:
        calib_data = json.load(f)
        camera_matrix = np.array(calib_data["camera_matrix"])
        dist_coeffs = np.array(calib_data["dist_coeff"])
except:
    print("[ERROR] camera_calibration.json not found! Exiting.")
    exit()

MARKER_SIZE = 0.21  # 3.5 cm
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
aruco_params = aruco.DetectorParameters()

CHECKPOINT_LOCATIONS = {
    # On the South-West corner of the large central debris pile
    0: {"x": 19.0, "y": 31.0, "name": "Central Debris Edge"},
    
    # On the South face of the right-hand scattered rubble chunk
    1: {"x": 36.0, "y": 27.0, "name": "Right Debris Chunk"},
    
    # On the South-East corner of the upper intersection chunk
    2: {"x": 34.0, "y": 43.0, "name": "Upper Intersection Debris"},
    
    # Dead center on the back wall of the arena for the final docking approach
    3: {"x": 26.0, "y": 83.0, "name": "Goal Wall Edge"}
}

CHECKPOINT_UDP_IP   = "127.0.0.1"
CHECKPOINT_UDP_PORT = 5005
_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# map scale is 10.
MAP_SCALE    = 10.0        
CAMERA_YAW_OFFSET = 0.0   

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def enhance_frame(frame):
    # This is heavy. We only call it right before YOLO now.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# ==========================================
# 4. MAIN LOOP
# ==========================================
stream = FastPiCamStream()
time.sleep(1.0) # Let the camera warm up and capture the first frame

frame_count = 0
yolo_boxes = None # Store the YOLO results to draw on frames in-between scans

print("[SYSTEM] Unified Radar Active. Press 'q' to quit.")

while True:
    frame = stream.latest_frame
    
    if frame is None:
        continue
        
    frame_count += 1
    display_frame = frame.copy() # Base frame for drawing
    
    # --------------------------------------------------
    # TASK A: YOLO HUMAN DETECTION (Every N Frames)
    # --------------------------------------------------
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # Only run CLAHE when we actually need it for YOLO
        enhanced_frame = enhance_frame(frame)
        results = model.predict(enhanced_frame, conf=0.25, imgsz=320, verbose=False)
        yolo_boxes = results[0] # Save the result object
        
    # Always draw the most recent YOLO boxes, even if we didn't run YOLO this exact frame
    if yolo_boxes is not None:
        display_frame = yolo_boxes.plot(img=display_frame)

    # --------------------------------------------------
    # TASK B: ARUCO CHECKPOINT LOCALIZATION (Every Frame)
    # --------------------------------------------------
    # ArUco runs on the raw, unenhanced frame (saves CPU, ArUco prefers raw contrast)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            aruco.drawDetectedMarkers(display_frame, corners)
            cv2.drawFrameAxes(display_frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

            x_m, y_m, z_m = tvecs[i][0][0], tvecs[i][0][1], tvecs[i][0][2]
            marker_id = ids[i][0]

            cv2.putText(display_frame, f"ID: {marker_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Z: {z_m*100:.1f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
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

stream.stop()
cv2.destroyAllWindows()
print("[SYSTEM] Live feed closed. Ready for next command.")