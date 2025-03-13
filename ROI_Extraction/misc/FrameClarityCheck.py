# import cv2
# import os
# import re

# def is_blurry(image_path, threshold=100):
#     """Check if an image is blurry using the Laplacian variance method."""
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
#     laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()  # Compute variance
#     return laplacian_var < threshold, laplacian_var  # Returns (is_blurry, score)

# def extract_numbers(filename):
#     """Extract the numeric parts from the filename to sort properly."""
#     # Use regular expressions to extract two numbers from the filename like 'frame_1_6.jpg'
#     match = re.match(r'frame_(\d+)_(\d+)', filename)
#     if match:
#         return int(match.group(1)), int(match.group(2))  # Return as a tuple (first_number, second_number)
#     return 0, 0  # Default if match fails

# # Path to frames folder
# frames_folder = "frames/video_2025-02-22_00-39-13"  # Change to your folder

# # List to store blurry images
# blurry_images = []

# # Check each frame (sorted sequentially by numeric part of filename)
# for filename in sorted(os.listdir(frames_folder), key=extract_numbers):
#     image_path = os.path.join(frames_folder, filename)
    
#     if filename.endswith(".jpg"):
#         blurry, score = is_blurry(image_path)
#         print(f"{filename} - Sharpness Score: {score} {'(Blurry)' if blurry else '(Clear)'}")
        
#         if blurry:
#             blurry_images.append(filename)  # Store blurry image names
#             # os.remove(image_path)  # Delete blurry image
#             # print(f"Deleted: {filename}")

# # Print all blurry images in sequential order
# if blurry_images:
#     print("\nBlurry Images Detected & Deleted:")
#     for img in blurry_images:
#         print(img)
# else:
#     print("\nNo blurry images detected.")

# print("Blurry image check completed.")


import cv2
import os
import re
import numpy as np

# def apply_superres(image):
#     """Use OpenCV's DNN super-resolution model to estimate image sharpness."""
#     # Load pre-trained DNN model for super-resolution
#     sr = cv2.dnn_superres.DnnSuperResImpl_create()
#     model_path = "models/EDSR_x3.pb"  # Path to pre-trained model, such as EDSR or any other
#     sr.readModel(model_path)
    
#     # Set the desired scale factor (e.g., 3x)
#     sr.setModel("edsr", 3)
    
#     # Apply the super-resolution model
#     result = sr.upsample(image)
#     return result

def apply_superres(image):
    """Use OpenCV's DNN super-resolution model to estimate image sharpness."""
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    model_path = "models/FSRCNN_x4.pb"  # Replace with FSRCNN model path
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 4)  # Set model type to "fsrcnn"
    result = sr.upsample(image)
    return result


def is_blurry(image_path, threshold=100):
    """Check if an image is blurry using Laplacian variance and super-res model."""
    image = cv2.imread(image_path)

    # Apply the super-resolution model to the image
    enhanced_image = apply_superres(image)
    
    # Calculate Laplacian variance for sharpness of the enhanced image
    gray_enhanced = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_enhanced, cv2.CV_64F).var()

    # If the model doesn't improve the sharpness, it's blurry
    return laplacian_var < threshold, laplacian_var

def extract_numbers(filename):
    """Extract the numeric parts from the filename to sort properly."""
    match = re.match(r'frame_(\d+)_(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0

# Path to frames folder
frames_folder = "frames/video_2025-02-22_00-39-13"  # Change to your folder

# List to store blurry images
blurry_images = []

# Check each frame (sorted sequentially by numeric part of filename)
for filename in sorted(os.listdir(frames_folder), key=extract_numbers):
    image_path = os.path.join(frames_folder, filename)
    
    if filename.endswith(".jpg"):
        blurry, score = is_blurry(image_path)
        print(f"{filename} - Sharpness Score: {score} {'(Blurry)' if blurry else '(Clear)'}")
        
        if blurry:
            blurry_images.append(filename)  # Store blurry image names
            # os.remove(image_path)  # Delete blurry image
            print(f"Deleted: {filename}")

# Print all blurry images in sequential order
if blurry_images:
    print("\nBlurry Images Detected & Deleted:")
    for img in blurry_images:
        print(img)
else:
    print("\nNo blurry images detected.")

print("Blurry image check completed.")


# import cv2
# import os
# import re
# import numpy as np

# def is_blurry(image_path, threshold=100, method="laplacian"):
#     """Check if an image is blurry using either Laplacian variance or FFT."""
#     try:
#         if method == "laplacian":
#             image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#             laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
#             return laplacian_var < threshold, laplacian_var
#         elif method == "fft":
#             image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#             f = np.fft.fft2(image)
#             fshift = np.fft.fftshift(f)
#             magnitude_spectrum = 20 * np.log(np.abs(fshift))
#             height, width = image.shape
#             center_x, center_y = width // 2, height // 2
#             radius = min(center_x, center_y) // 3
#             mask = np.zeros_like(image, dtype=np.uint8)
#             cv2.circle(mask, (center_x, center_y), radius, 255, -1)
#             masked_magnitude = np.ma.masked_array(magnitude_spectrum, mask=~mask.astype(bool))
#             average_high_freq = np.mean(masked_magnitude)
#             return average_high_freq < 20, average_high_freq #adjust 20 as needed.
#         else:
#             raise ValueError("Invalid method. Choose 'laplacian' or 'fft'.")
#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#         return True, 0  # Treat as blurry if error occurs

# def extract_numbers(filename):
#     """Extract the numeric parts from the filename to sort properly."""
#     match = re.match(r'frame_(\d+)_(\d+)', filename)
#     if match:
#         return int(match.group(1)), int(match.group(2))
#     return 0, 0

# # Path to frames folder
# frames_folder = "frames/video_2025-02-22_00-39-13"  # Change to your folder

# # List to store blurry images
# blurry_images = []

# # Choose method 'laplacian' or 'fft'
# blur_detection_method = "laplacian" #or "fft"

# # Check each frame (sorted sequentially by numeric part of filename)
# for filename in sorted(os.listdir(frames_folder), key=extract_numbers):
#     image_path = os.path.join(frames_folder, filename)

#     if filename.endswith(".jpg"):
#         blurry, score = is_blurry(image_path, method=blur_detection_method)
#         print(f"{filename} - Sharpness Score: {score} {'(Blurry)' if blurry else '(Clear)'}")

#         if blurry:
#             blurry_images.append(filename)
#             # os.remove(image_path)  # Delete blurry image
#             # print(f"Deleted: {filename}")

# # Print all blurry images in sequential order
# if blurry_images:
#     print("\nBlurry Images Detected:")
#     for img in blurry_images:
#         print(img)
# else:
#     print("\nNo blurry images detected.")

# print("Blurry image check completed.")

# import cv2
# import os
# import re

# def is_in_focus(image_path, threshold=100):
#     """Checks if an image is in focus using the Laplacian variance."""
#     try:
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
#         return laplacian_var >= threshold, laplacian_var #Greater than or equal to.
#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#         return False, 0  # Treat as out of focus if error occurs

# def extract_numbers(filename):
#     """Extract numeric parts for sorting."""
#     match = re.match(r'frame_(\d+)_(\d+)', filename)
#     if match:
#         return int(match.group(1)), int(match.group(2))
#     return 0, 0

# # Path to frames folder
# frames_folder = "frames/video_2025-02-22_00-39-13"  # Change to your folder

# # List to store in-focus images
# in_focus_images = []

# # Check each frame (sorted sequentially)
# for filename in sorted(os.listdir(frames_folder), key=extract_numbers):
#     image_path = os.path.join(frames_folder, filename)

#     if filename.endswith(".jpg"):
#         in_focus, score = is_in_focus(image_path)
#         print(f"{filename} - Sharpness Score: {score} {'(In Focus)' if in_focus else '(Out of Focus)'}")

#         if in_focus:
#             in_focus_images.append(filename)

# # Print in-focus images
# if in_focus_images:
#     print("\nIn-Focus Images Detected:")
#     for img in in_focus_images:
#         print(img)
# else:
#     print("\nNo in-focus images detected.")

# print("In-focus image check completed.")