import os
import cv2
from ultralytics import YOLO

def enhance_frame(frame):
    """
    Applies CLAHE to the image matrix (frame) to improve contrast in dark/dusty disaster rubble.
    Modified to accept a raw frame directly to support rapid video stream processing.
    """
    if frame is None:
        return None
        
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def main():
    print("[SYSTEM] Booting Multi-Modal YOLO11 Kinematic Pose Estimation Protocol...")

    # Load the state-of-the-art YOLO11 Large Pose model
    model = YOLO('yolo11l-pose.pt')
    
    dataset_path = "test_dataset"
    output_dir = "detection_results"
    os.makedirs(output_dir, exist_ok=True)

    # Define supported sensory file extensions
    image_exts = ('.png', '.jpg', '.jpeg')
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')

    for file_name in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file_name)
        file_lower = file_name.lower()
        
        # ---------------------------------------------------------
        # PROCESS STATIC OPTICAL FEEDS (IMAGES)
        # ---------------------------------------------------------
        if file_lower.endswith(image_exts):
            print(f"[VISION] Analyzing static skeletal topology in {file_name}...")
            frame = cv2.imread(file_path)
            enhanced_frame = enhance_frame(frame)
            
            if enhanced_frame is not None:
                results = model.predict(enhanced_frame, conf=0.25)
                
                for result in results:
                    annotated_frame = result.plot()
                    output_path = os.path.join(output_dir, f"yolo11_pose_{file_name}")
                    cv2.imwrite(output_path, annotated_frame)
                    print(f"[SYSTEM] Static pose topology saved to: {output_path}")

        # ---------------------------------------------------------
        # PROCESS DYNAMIC OPTICAL FEEDS (VIDEOS)
        # ---------------------------------------------------------
        elif file_lower.endswith(video_exts):
            print(f"[VISION] Initializing continuous video stream analysis on {file_name}...")
            cap = cv2.VideoCapture(file_path)
            
            # Extract spatial constraints and framerate for the video writer
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Handle potential zero-FPS read errors
            if fps == 0: fps = 30 
                
            output_path = os.path.join(output_dir, f"yolo11_pose_{file_name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 compilation
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # Visual stream ended
                
                frame_count += 1
                
                # Output telemetry every 30 frames to monitor progress without console flooding
                if frame_count % 30 == 0:
                    print(f"[SYSTEM] Processing frame {frame_count}/{total_frames} of {file_name}...")
                    
                enhanced_frame = enhance_frame(frame)
                
                if enhanced_frame is not None:
                    # verbose=False mutes YOLO's per-frame terminal logging for performance
                    results = model.predict(enhanced_frame, conf=0.25, verbose=False)
                    
                    # YOLO predict on a single frame returns a list with one item
                    annotated_frame = results[0].plot()
                    
                    # Write the frame back into the output video stream
                    out.write(annotated_frame)
                    
            cap.release()
            out.release()
            print(f"[SYSTEM] Dynamic continuous tracking saved to: {output_path}")

    print("[SYSTEM] Scan complete. Ready to calculate Joint 1 & Joint 2 rotations based on extracted keypoint coordinates.")

if __name__ == "__main__":
    main()