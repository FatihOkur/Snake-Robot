import os
import cv2
from ultralytics import YOLO

def main():
    # As the Autonomous Modular Snake Robot, initializing the vision module
    print("[SYSTEM] Initializing YOLO-World Vision System for Search and Rescue...")

    # Load a pre-trained YOLOv8-World model (Small version is good for real-time robotic processing)
    # This model downloads automatically on the first run.
    model = YOLO('yolov8s-world.pt')

    # Define the specific classes we are looking for in the disaster environment
    # YOLO-World uses open-vocabulary, so it understands these text prompts directly.
    custom_classes = ["person", "human face", "human hand", "arm", "leg"]
    model.set_classes(custom_classes)
    
    print(f"[SYSTEM] Target classes set to: {custom_classes}")

    # Path to the test dataset based on your directory structure
    dataset_path = "test_dataset"
    output_dir = "detection_results"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Verify the test dataset exists
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path not found: {dataset_path}")
        return

    # Process each image in the dataset
    for img_name in os.listdir(dataset_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dataset_path, img_name)
            print(f"[VISION] Scanning {img_name}...")
            
            # Run inference on the image
            # conf=0.05 is the confidence threshold. In disaster scenarios, we might want 
            # a lower threshold to ensure we don't miss partially buried victims.
            results = model.predict(img_path, conf=0.15)
            
            # Save and display the results
            for result in results:
                # Generate the annotated image matrix
                annotated_frame = result.plot()
                
                # Save the image to the output directory
                output_path = os.path.join(output_dir, f"detected_{img_name}")
                cv2.imwrite(output_path, annotated_frame)
                print(f"[SYSTEM] Saved detection results to: {output_path}")

    print("[SYSTEM] Scan complete. Awaiting movement commands for Joint 1 to acquire new visual field.")

if __name__ == "__main__":
    main()