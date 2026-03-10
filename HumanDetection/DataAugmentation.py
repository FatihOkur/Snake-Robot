import cv2
import numpy as np
import os
import glob
from skimage import exposure

def add_camera_noise(image, intensity=15.0):
    """
    Injects synthetic Gaussian sensor noise into the image to simulate 
    the low-light electrical noise of the OV5647 sensor.
    """
    row, col, ch = image.shape
    mean = 0
    sigma = intensity
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def transform_dataset(input_dir, output_dir, reference_image_path):
    """
    Transforms clean RGB images into the OV5647 IR-CUT camera style.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the reference image (the image you obtained from the camera)
    reference_img = cv2.imread(reference_image_path)
    if reference_img is None:
        print(f"ERROR: Could not load reference image at {reference_image_path}")
        return

    # Find all JPGs in the input directory (you can add .png etc. if needed)
    search_pattern = os.path.join(input_dir, '*.jpg')
    image_paths = glob.glob(search_pattern)

    if len(image_paths) == 0:
        print(f"No images found in {input_dir}. Please check the folder path.")
        return

    print(f"Found {len(image_paths)} images to process. Initializing transformation pipeline...")

    for img_path in image_paths:
        # 1. Read the source image
        source_img = cv2.imread(img_path)
        if source_img is None:
            continue
            
        # 2. Downscale/Resize (Simulating 640x640 resolution for edge-device YOLO training)
        source_img = cv2.resize(source_img, (640, 640))
        ref_resized = cv2.resize(reference_img, (640, 640))
        
        # 3. Histogram Matching 
        # (This mathematically transfers the specific IR tint, contrast, and brightness from your camera)
        source_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        ref_rgb = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2RGB)
        
        matched_rgb = exposure.match_histograms(source_rgb, ref_rgb, channel_axis=-1)
        matched_bgr = cv2.cvtColor(matched_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # 4. Inject Low-Light Sensor Noise
        noisy_img = add_camera_noise(matched_bgr, intensity=25.0) # Adjust intensity if needed
        
        # 5. Add Slight Lens Softness/Blur (Simulating the 3.6mm lens focus falloff)
        final_img = cv2.GaussianBlur(noisy_img, (3, 3), 0)
        
        # 6. Save the transformed image
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"IR_synthetic_{filename}")
        cv2.imwrite(output_path, final_img)
        
    print(f"Pipeline complete! {len(image_paths)} transformed images saved to {output_dir}.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Place your downloaded COCO or Pascal VOC images in this folder
    INPUT_DATASET_FOLDER = "rgb_dataset_input" 
    
    # The processed images will appear here
    OUTPUT_DATASET_FOLDER = "ir_dataset_output/train" 
    
    # The path to the image you uploaded
    REFERENCE_IMAGE = "image_002.jpg" 
    
    # Create input folder if it doesn't exist so you can drop images in it
    if not os.path.exists(INPUT_DATASET_FOLDER):
        os.makedirs(INPUT_DATASET_FOLDER)
        print(f"Created folder '{INPUT_DATASET_FOLDER}'. Please place some standard RGB images inside and run the script again.")
    else:
        # Run the pipeline
        transform_dataset(INPUT_DATASET_FOLDER, OUTPUT_DATASET_FOLDER, REFERENCE_IMAGE)