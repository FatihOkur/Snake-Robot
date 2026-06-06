import cv2
from ultralytics import YOLO
import numpy as np

def enhance_frame(frame):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    in the LAB color space to improve visibility in dark debris environments.
    """
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
    
    # 2. GSTREAMER PIPELINE (Optimized for Raspberry Pi 5 & libcamera)
    # Explicitly formatted to BGR to prevent OpenCV 'reshape' errors, 
    # and using drop=true sync=false to prevent camera stalling.
    gstreamer_pipeline = (
        "libcamerasrc ! "
        "video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true sync=false"
    )
    
    print("[VISION] Initializing camera interface...")
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Ensure the ribbon cable is secure and libcamera is functioning.")
        return

    # Configurable frame skip for Raspberry Pi (Process 1 frame out of every N frames)
    PROCESS_EVERY_N_FRAMES = 3 
    frame_count = 0
    display_frame = None

    print("[SYSTEM] Scan complete. Live feed starting. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        
        # 3. ROBUST FRAME VALIDATION
        # Prevents crash if the camera buffer drops a frame or sends empty data
        if not ret or frame is None or frame.size == 0:
            print("[WARN] Received empty frame, skipping...")
            continue
            
        frame_count += 1
        
        # Apply dark-environment enhancements
        enhanced_frame = enhance_frame(frame)
        
        # 4. FRAME SKIPPING: Only run heavy AI on every Nth frame
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # Run inference at lower resolution (imgsz=320) to save CPU
            results = model.predict(enhanced_frame, conf=0.25, imgsz=320, verbose=False)
            display_frame = results[0].plot()
        elif display_frame is None:
            # If we haven't processed a frame yet, just pass the raw enhanced frame
            display_frame = enhanced_frame
            
        # Display the live output
        cv2.imshow('Snake Robot - Live Human Detection', display_frame)
        
        # Break the loop safely if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[SYSTEM] User requested shutdown...")
            break

    # 5. SAFE CLEANUP
    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Live feed closed. Ready for next command.")

if __name__ == "__main__":
    main()