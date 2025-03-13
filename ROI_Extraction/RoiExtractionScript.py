# import os
# import subprocess
# import shutil
# from datetime import datetime
# import cv2  # Ensure OpenCV is installed: pip install opencv-python
# import re
# import matplotlib.pyplot as plt  # For displaying images in Jupyter
# import itertools
# import numpy as np
# from tqdm import tqdm
# import glob
# from RoiExtraction import *
# import csv
# import json
# import pandas as pd

# # Create main Img directory and subdirectories for the different image types
# img_dir = os.path.join(os.getcwd(), "Img")
# captured_dir = os.path.join(img_dir, "CameraInput")
# roi_dir = os.path.join(img_dir, "ROI")
# annotation_dir = os.path.join(img_dir, "Annotations")

# os.makedirs(captured_dir, exist_ok=True)
# os.makedirs(roi_dir, exist_ok=True)
# os.makedirs(annotation_dir, exist_ok=True)

# # Define Darknet parameters
# darknet_dir = os.path.join(os.getcwd(), "darknet")

# ###################################################################################
# # Path to the obj.data file for Darknet -> Update this path as needed
# obj_data_path = "/Users/marcusfoo/Desktop/ResearchProject/ROI_Extraction/obj.data"


# ###################################################################################

# cfg_file = os.path.join(os.getcwd(),"yolov3-tiny.cfg")
# weights_file = os.path.join(os.getcwd(),"yolov3-tiny_final.weights")
# supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')

# # --- Capture an Image from the Camera with User Input ---
# cap = cv2.VideoCapture(1)  # 0 for default camera
# if not cap.isOpened():
#     print("Error: Could not access the camera.")
# else:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture image from camera.")
#             break

#         # Display the camera feed
#         cv2.imshow("Camera Feed - Press 'c' to capture, 'q' to quit", frame)
#         key = cv2.waitKey(1) & 0xFF

#         # If the user presses 'c', capture the frame
#         if key == ord('c'):
#             input_image_path = os.path.join(captured_dir, f"camera_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
#             cv2.imwrite(input_image_path, frame)
#             print("Captured image from camera and saved to:", input_image_path)
#             break
#         # Optionally, allow quitting without capturing
#         elif key == ord('q'):
#             print("Exiting without capturing an image.")
#             cap.release()
#             cv2.destroyAllWindows()
#             exit()

#     cap.release()
#     cv2.destroyAllWindows()

# image = cv2.imread(input_image_path)
# if image is None:
#     print("Error: Could not read the captured image.")
# else:
#     # Run Darknet detection on the captured image
#     detections = run_darknet_detection(
#         darknet_dir=darknet_dir,
#         obj_data_path=obj_data_path,
#         cfg_file=cfg_file,
#         weights_file=weights_file,
#         input_image=input_image_path,
#         output_dir=captured_dir,
#     )
    
#     # Extract detection points from the detections
#     center_points = extract_points_from_detections(detections)
#     if len(center_points) < 4:
#         print("Not enough points detected in the camera image. Exiting detection.")
#     else:
#         # Find the best trio of points and the thumb-index gap
#         closest_trio, thumb_index_gap = find_closest_trio(center_points)
#         if len(closest_trio) < 3 or thumb_index_gap is None:
#             print("Could not find a suitable trio in the camera image. Exiting detection.")
#         else:
#             # Calculate midpoints and point C for ROI extraction
#             midpoints = calculate_midpoints(closest_trio, thumb_index_gap)
#             point_c = calculate_point_c(midpoints[0], midpoints[1], thumb_index_gap=thumb_index_gap)
#             # Here, hand type can be set as needed (defaulting to 'right' for this example)
#             hand_type = 'right'
#             roi, box = extract_roi(image, midpoints, point_c, thumb_index_gap, hand_type=hand_type)
            
#             # Save the ROI image in the ROI folder
#             base_name = os.path.splitext(os.path.basename(input_image_path))[0]
#             roi_image_name = f"{base_name}_ROI.jpg"
#             roi_destination_path = os.path.join(roi_dir, roi_image_name)
#             cv2.imwrite(roi_destination_path, roi)
#             print("ROI image saved to:", roi_destination_path)
            
#             # Annotate the original image with detection details
#             annotated_image = image.copy()
#             for idx, point in enumerate(closest_trio):
#                 cv2.circle(annotated_image, (int(point[0]), int(point[1])), 10, (255, 0, 0), -1)
#                 cv2.putText(annotated_image, f"P{idx+1}", (int(point[0]) + 5, int(point[1]) - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#             cv2.circle(annotated_image, (int(thumb_index_gap[0]), int(thumb_index_gap[1])), 10, (0, 255, 0), -1)
#             cv2.putText(annotated_image, "Thumb-Index Gap", (int(thumb_index_gap[0]) + 5, int(thumb_index_gap[1]) - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             cv2.circle(annotated_image, (int(midpoints[0][0]), int(midpoints[0][1])), 10, (0, 0, 255), -1)
#             cv2.circle(annotated_image, (int(midpoints[1][0]), int(midpoints[1][1])), 10, (0, 0, 255), -1)
#             cv2.circle(annotated_image, (int(point_c[0]), int(point_c[1])), 10, (255, 255, 0), -1)
#             cv2.putText(annotated_image, "Point C", (int(point_c[0]) + 5, int(point_c[1]) - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
#             cv2.drawContours(annotated_image, [box], 0, (0, 255, 0), 2)
            
#             # Save the annotated image in the Annotations folder
#             annotated_image_name = f"{base_name}_annotated.jpg"
#             annotated_image_path = os.path.join(annotation_dir, annotated_image_name)
#             cv2.imwrite(annotated_image_path, annotated_image)
#             print("Annotated image saved to:", annotated_image_path)

# print("Camera image detection and ROI extraction complete.")

# # --- Capture an Image from the Camera with User Input ---

import sys
import os
import cv2
import glob
from datetime import datetime
from RoiExtraction import *  


if len(sys.argv) < 2:
    print("Usage: python ROIExtractionScript.py <image_folder>")
    sys.exit(1)

image_folder = sys.argv[1]
print(f"Processing images from: {image_folder}")

# Define directories
input_dir = os.path.join(os.getcwd(), image_folder)  # Folder containing images to process
output_root = os.path.join(os.getcwd(), "ProcessedData")  # Root folder for processed data

# Get the folder name from the input directory
folder_name = os.path.basename(input_dir)

# Create output directory structure based on the folder name
roi_dir = os.path.join(output_root, "ROI", folder_name)
annotation_dir = os.path.join(output_root, "Annotations", folder_name)
os.makedirs(roi_dir, exist_ok=True)
os.makedirs(annotation_dir, exist_ok=True)

# Darknet parameters (update these paths as needed)
darknet_dir = os.path.join(os.getcwd(), "darknet")
obj_data_path = "/Users/marcusfoo/Documents/GitHub/palmprint-authenticator/ROI_Extraction/obj.data"
cfg_file = os.path.join(os.getcwd(), "yolov3-tiny.cfg")
weights_file = os.path.join(os.getcwd(), "yolov3-tiny_final.weights")
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')

# Get all images in the input folder
image_paths = glob.glob(os.path.join(input_dir, "*"))
image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in supported_formats]

if not image_paths:
    print(f"No images found in {input_dir} with supported formats: {supported_formats}")
else:
    for input_image_path in image_paths:
        print(f"Processing image: {input_image_path}")
        
        # Load the image
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Error: Could not read the image: {input_image_path}")
            continue

        # Run Darknet detection on the image
        detections = run_darknet_detection(
            darknet_dir=darknet_dir,
            obj_data_path=obj_data_path,
            cfg_file=cfg_file,
            weights_file=weights_file,
            input_image=input_image_path,
            output_dir=os.path.dirname(input_image_path),
        )

        # Extract detection points from the detections
        center_points = extract_points_from_detections(detections)
        if len(center_points) < 4:
            print(f"Not enough points detected in {input_image_path}. Skipping.")
            continue

        # Find the best trio of points and the thumb-index gap
        closest_trio, thumb_index_gap = find_closest_trio(center_points)
        if len(closest_trio) < 3 or thumb_index_gap is None:
            print(f"Could not find a suitable trio in {input_image_path}. Skipping.")
            continue

        # Calculate midpoints and point C for ROI extraction
        midpoints = calculate_midpoints(closest_trio, thumb_index_gap)
        point_c = calculate_point_c(midpoints[0], midpoints[1], thumb_index_gap=thumb_index_gap)

        # Extract ROI (defaulting to 'right' hand)
        hand_type = 'right'
        roi, box = extract_roi(image, midpoints, point_c, thumb_index_gap, hand_type=hand_type)

        # Save the ROI image
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        roi_image_name = f"{base_name}_ROI.jpg"
        roi_destination_path = os.path.join(roi_dir, roi_image_name)
        cv2.imwrite(roi_destination_path, roi)
        print(f"ROI image saved to: {roi_destination_path}")

        # Annotate the original image with detection details
        annotated_image = image.copy()
        for idx, point in enumerate(closest_trio):
            cv2.circle(annotated_image, (int(point[0]), int(point[1])), 10, (255, 0, 0), -1)
            cv2.putText(annotated_image, f"P{idx+1}", (int(point[0]) + 5, int(point[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(annotated_image, (int(thumb_index_gap[0]), int(thumb_index_gap[1])), 10, (0, 255, 0), -1)
        cv2.putText(annotated_image, "Thumb-Index Gap", (int(thumb_index_gap[0]) + 5, int(thumb_index_gap[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(annotated_image, (int(midpoints[0][0]), int(midpoints[0][1])), 10, (0, 0, 255), -1)
        cv2.circle(annotated_image, (int(midpoints[1][0]), int(midpoints[1][1])), 10, (0, 0, 255), -1)
        cv2.circle(annotated_image, (int(point_c[0]), int(point_c[1])), 10, (255, 255, 0), -1)
        cv2.putText(annotated_image, "Point C", (int(point_c[0]) + 5, int(point_c[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.drawContours(annotated_image, [box], 0, (0, 255, 0), 2)

        # Save the annotated image
        annotated_image_name = f"{base_name}_annotated.jpg"
        annotated_image_path = os.path.join(annotation_dir, annotated_image_name)
        cv2.imwrite(annotated_image_path, annotated_image)
        print(f"Annotated image saved to: {annotated_image_path}")

print("Batch processing complete.")

