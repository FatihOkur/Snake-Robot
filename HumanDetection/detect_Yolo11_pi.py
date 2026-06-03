import os
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
    
    dataset_path = "test_dataset"
    output_dir = "detection_results_pi"
    os.makedirs(output_dir, exist_ok=True)

    image_exts = ('.png', '.jpg', '.jpeg')
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')

    # Configurable frame skip for Raspberry Pi (Process 1 frame out of every N frames)
    # Increase this number if the Pi is still lagging.
    PROCESS_EVERY_N_FRAMES = 3 

    for file_name in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file_name)
        file_lower = file_name.lower()
        
        # --- IMAGE PROCESSING ---
        if file_lower.endswith(image_exts):
            print(f"[VISION] Analyzing {file_name}...")
            frame = cv2.imread(file_path)
            enhanced_frame = enhance_frame(frame)
            
            if enhanced_frame is not None:
                # 2. REDUCE RESOLUTION (imgsz=320) to speed up Pi processing
                results = model.predict(enhanced_frame, conf=0.25, imgsz=320)
                for result in results:
                    cv2.imwrite(os.path.join(output_dir, f"pi_pose_{file_name}"), result.plot())

        # --- VIDEO PROCESSING ---
        elif file_lower.endswith(video_exts):
            print(f"[VISION] Starting optimized video analysis on {file_name}...")
            cap = cv2.VideoCapture(file_path)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0: fps = 30 
                
            out = cv2.VideoWriter(os.path.join(output_dir, f"pi_pose_{file_name}"), 
                                  cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            frame_count = 0
            last_annotated_frame = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                enhanced_frame = enhance_frame(frame)
                
                # 3. FRAME SKIPPING: Only run heavy AI on every Nth frame
                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    # Run inference at lower resolution (imgsz=320)
                    results = model.predict(enhanced_frame, conf=0.25, imgsz=320, verbose=False)
                    last_annotated_frame = results[0].plot()
                elif last_annotated_frame is None:
                    # If we haven't processed a frame yet, just pass the raw frame
                    last_annotated_frame = enhanced_frame
                
                # Write the most recently calculated frame to keep the video smooth
                out.write(last_annotated_frame)
                    
            cap.release()
            out.release()
            print(f"[SYSTEM] Optimized video saved to: {output_dir}/pi_pose_{file_name}")

    print("[SYSTEM] Scan complete. Ready for kinematic execution.")

if __name__ == "__main__":
    main()