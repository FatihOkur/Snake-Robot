import os
import cv2
from ultralytics import YOLO

def enhance_optical_feed(image_path):
    """
    Applies CLAHE to the image to improve contrast in dark/dusty disaster rubble.
    This helps the model distinguish human skin/clothing from debris.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Convert to LAB color space to isolate illumination (Lightness)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    
    # Merge channels and convert back to BGR
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

def main():
    print("[SYSTEM] Booting Enhanced YOLO-World Vision Protocol...")

    # 1. UPGRADE MODEL: Switching to the Large model for higher accuracy
    model = YOLO('yolov8l-world.pt')

    # 2. OPTIMIZE PROMPTS: Simplified vocabulary often yields higher confidence
    custom_classes = ["person", "face", "hand", "arm", "leg", "foot"]
    model.set_classes(custom_classes)
    
    print(f"[SYSTEM] Calibrated target parameters: {custom_classes}")

    dataset_path = "test_dataset"
    output_dir = "detection_results"
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(dataset_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dataset_path, img_name)
            print(f"[VISION] Processing {img_name}...")
            
            # 3. ENHANCE CONTRAST
            enhanced_frame = enhance_optical_feed(img_path)
            
            if enhanced_frame is not None:
                # Run inference directly on the enhanced numpy array
                # We can keep conf=0.10 or 0.15; the model size + CLAHE will boost the actual scores
                results = model.predict(enhanced_frame, conf=0.15)
                
                for result in results:
                    annotated_frame = result.plot()
                    output_path = os.path.join(output_dir, f"enhanced_detected_{img_name}")
                    cv2.imwrite(output_path, annotated_frame)
                    print(f"[SYSTEM] Target data saved: {output_path}")

    print("[SYSTEM] Scan complete. Vision logic standing by.")

if __name__ == "__main__":
    main()