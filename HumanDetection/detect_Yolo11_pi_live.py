import cv2
from ultralytics import YOLO

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

    # 1. DOWNGRADE TO NANO MODEL for SBC (Single Board Computer) compatibility
    model = YOLO('yolo11n-pose.pt')
    
    # Configurable frame skip for Raspberry Pi (Process 1 frame out of every N frames)
    # Increase this number if the Pi is still lagging.
    PROCESS_EVERY_N_FRAMES = 3 

    print("[VISION] Starting live camera feed...")
    # Camera Initialization
    # Note: Raspberry Pi 5 users might need to replace 0 with a GStreamer pipeline if libcamera causes issues.
    cap = cv2.VideoCapture(0)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera.")
            break
        
        frame_count += 1
        enhanced_frame = enhance_frame(frame)
        
        # 3. FRAME SKIPPING: Only run heavy AI on every Nth frame
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # Run inference at lower resolution (imgsz=320)
            results = model.predict(enhanced_frame, conf=0.25, imgsz=320, verbose=False)
            display_frame = results[0].plot()
        else:
            # On skipped frames, simply display the enhanced_frame to maintain a smooth video output
            display_frame = enhanced_frame
        
        # Display the frame
        cv2.imshow("YOLO11 Pose Estimation - Live", display_frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Live feed closed.")

if __name__ == "__main__":
    main()
