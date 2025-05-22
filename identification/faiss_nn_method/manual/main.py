import os
import json
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# -------------------- Configuration -------------------- #
DATA_DIR = "data"  # Folder where your data files are stored
FEATURE_MATRIX_PATH = os.path.join(DATA_DIR, "feature_matrix_s14_no_interpolate.npy")
CLASS_DICT_PATH = os.path.join(DATA_DIR, "feature_matrix_s14_no_interpolate.json")
NEW_MATRIX_PATH = os.path.join(DATA_DIR, "test.npy")

# -------------------- Model & Transformation Setup -------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()

print("Loading DINOv2 model...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
model.eval()
model.to(device)
if device.type == "cuda":
    model.half()
print("DINOv2 model loaded on device:", device)

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# -------------------- Helper Functions -------------------- #
def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

def extract_features(image_path):
    try:
        pil_img = Image.open(image_path).convert("L")
        # Replicate grayscale channel to create a 3-channel image.
        pil_img = Image.merge("RGB", (pil_img, pil_img, pil_img))
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    if device.type == "cuda":
        img_tensor = img_tensor.half()
    with torch.no_grad():
        features = model(img_tensor)
    features = features.cpu().numpy().flatten()
    return normalize_vector(features)

# -------------------- Load Gallery and Class Dictionary -------------------- #
if os.path.exists(NEW_MATRIX_PATH):
    A = np.load(NEW_MATRIX_PATH)
    print(f"Loaded feature matrix with shape: {A.shape}")
elif os.path.exists(FEATURE_MATRIX_PATH):
    A = np.load(FEATURE_MATRIX_PATH)
    print(f"Loaded feature matrix with shape: {A.shape}")
else:
    # For DINOv2_vits14, we expect 384-dimensional features.
    A = np.empty((384, 0))
    np.save(FEATURE_MATRIX_PATH, A)
    print(f"Created new feature matrix with shape: {A.shape}")

if os.path.exists(CLASS_DICT_PATH):
    with open(CLASS_DICT_PATH, "r") as f:
        class_dict = json.load(f)
else:
    class_dict = {}

# -------------------- Main Loop -------------------- #
while True:
    mode = input("Enter mode (enroll / identify) or 'quit' to exit: ").strip().lower()
    if mode == "quit":
        print("Exiting program.")
        break
    elif mode not in ["enroll", "identify"]:
        print("Invalid mode. Please enter 'enroll', 'identify', or 'quit'.")
        continue

    if mode == "enroll":
        from enroll import run_enrollment
        A, class_dict = run_enrollment(A, class_dict, NEW_MATRIX_PATH, extract_features)
        np.save(NEW_MATRIX_PATH, A)
        with open(CLASS_DICT_PATH, "w") as f:
            json.dump(class_dict, f, indent=4)
    elif mode == "identify":
        from identify import run_identification
        run_identification(A, class_dict, extract_features)
    
    print("\n--- Operation complete. ---\n")
