import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import json

# Load the existing feature matrix
feature_matrix_path = "feature_matrix_A.npy"
class_dict_path = "class_dict.json"  # Assuming you have stored class info separately

if os.path.exists(feature_matrix_path):
    A = np.load(feature_matrix_path)
    print(f"Loaded feature matrix with shape: {A.shape}")
else:
    print("Feature matrix not found. Exiting.")
    exit(1)

# Load the class dictionary
if os.path.exists(class_dict_path):
    with open(class_dict_path, 'r') as f:   
        class_dict = json.load(f)
else:
    class_dict = {}

# -------------------- DINOv2 Model Setup -------------------- #
print("Loading DINOv2 model...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
model.eval()
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"DINOv2 model loaded on device: {device}")

# Define image transformation pipeline
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -------------------- Feature Extraction for New Participant -------------------- #
def normalize_vector(vector):
    """
    Normalize a vector to unit norm.
    """
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

def extract_features(image_path):
    """
    Extract a feature vector from an image using DINOv2.
    """
    try:
        pil_img = Image.open(image_path).convert("L")
        # Convert grayscale to 3-channel RGB
        pil_img = Image.merge("RGB", (pil_img, pil_img, pil_img))
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img_tensor).cpu().numpy().flatten()
    return normalize_vector(features)

# Provide new participant images (update this with actual file paths)
new_participant_id = "201"  # Example participant ID
new_images = [
    "/home/nemo/Documents/palmprint-authenticator/ROI_Extraction/Img/ROI/camera_input_20250221_230525_ROI.jpg", 
    "/home/nemo/Documents/palmprint-authenticator/ROI_Extraction/Img/ROI/camera_input_20250222_002153_ROI.jpg",
    "/home/nemo/Documents/palmprint-authenticator/ROI_Extraction/Img/ROI/camera_input_20250222_002820_ROI.jpg",
    "/home/nemo/Documents/palmprint-authenticator/ROI_Extraction/Img/ROI/camera_input_20250222_121229_ROI.jpg"
]  # Replace with actual image paths

# Extract features for new images
new_features_list = []
for img_path in new_images:
    feature_vector = extract_features(img_path)
    if feature_vector is not None:
        new_features_list.append(feature_vector)

if len(new_features_list) == 0:
    print("No valid images found for new participant. Exiting.")
    exit(1)

# Convert list to numpy array and adjust shape
new_features = np.array(new_features_list)  # shape: (num_images, feature_dim)
print("new_features shape before transpose:", new_features.shape)

# Transpose so each column represents a feature vector
new_features = new_features.T  # shape: (feature_dim, num_images)
print("new_features shape after transpose:", new_features.shape)

# Pad or truncate new_features to have 1024 rows (to match A's number of rows)
if new_features.shape[0] < 1024:
    pad_rows = 1024 - new_features.shape[0]
    new_features = np.vstack((new_features, np.zeros((pad_rows, new_features.shape[1]))))
elif new_features.shape[0] > 1024:
    new_features = new_features[:1024, :]

print("new_features shape after padding/truncation:", new_features.shape)

# -------------------- Expand Feature Matrix -------------------- #
print(f"Original feature matrix shape: {A.shape}")
A_expanded = np.hstack((A, new_features))  # Append new features as new columns
np.save(feature_matrix_path, A_expanded)
print(f"Updated feature matrix shape: {A_expanded.shape}")

# -------------------- Update Class Dictionary -------------------- #
start_index = A.shape[1]  # New feature index starts at the previous matrix size
new_indices = list(range(start_index, start_index + new_features.shape[1]))
class_dict[new_participant_id] = new_indices

# Save updated class dictionary
with open(class_dict_path, 'w') as f:
    json.dump(class_dict, f, indent=4)

print(f"New participant {new_participant_id} enrolled successfully!")
