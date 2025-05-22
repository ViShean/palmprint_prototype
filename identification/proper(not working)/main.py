import os
import json
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import sys

# -------------------- Configuration -------------------- #
DATA_DIR = "data"  # Folder where data files are stored
FEATURE_MATRIX_PATH = os.path.join(DATA_DIR, "feature_matrix_s14_no_interpolate.npy")
CLASS_DICT_PATH = os.path.join(DATA_DIR, "feature_matrix_s14_no_interpolate.json")
NEW_MATRIX_PATH = os.path.join(DATA_DIR, "test.npy")
FACTOR_FILE = os.path.join(DATA_DIR, "gallery_factorization.pt")

LAMBDA_REG = 0.001  # Regularization parameter

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

def precompute_gallery_factorization(A, lambda_=LAMBDA_REG):
    """Compute the full Cholesky factorization of (A^T A + lambda I)."""
    A_torch = torch.tensor(A, dtype=torch.float32, device=device)
    At = A_torch.t()
    AtA = torch.matmul(At, A_torch)
    n = A_torch.shape[1]
    I = torch.eye(n, device=device, dtype=torch.float32)
    regularized_matrix = AtA + lambda_ * I
    L = torch.linalg.cholesky(regularized_matrix)
    return L

# -------------------- Main Loop -------------------- #
while True:
    mode = input("Enter mode (enroll / identify) or 'quit' to exit: ").strip().lower()
    if mode == "quit":
        print("Exiting program.")
        break
    elif mode not in ["enroll", "identify"]:
        print("Invalid mode. Please enter 'enroll', 'identify', or 'quit'.")
        continue

    # Load or create feature matrix A
    if os.path.exists(NEW_MATRIX_PATH):
        A = np.load(NEW_MATRIX_PATH)
        print(f"Loaded feature matrix with shape: {A.shape}")
    elif os.path.exists(FEATURE_MATRIX_PATH):
        A = np.load(FEATURE_MATRIX_PATH)
        print(f"Loaded feature matrix with shape: {A.shape}")
    else:
        A = np.empty((384, 0))  # assuming 384-dimensional features for dinov2_vits14
        np.save(FEATURE_MATRIX_PATH, A)
        print(f"Created new feature matrix with shape: {A.shape}")

    # Load class dictionary
    if os.path.exists(CLASS_DICT_PATH):
        with open(CLASS_DICT_PATH, 'r') as f:
            class_dict = json.load(f)
    else:
        class_dict = {}

    if mode == "enroll":
        from enroll import run_enrollment
        A, class_dict = run_enrollment(A, class_dict, NEW_MATRIX_PATH, extract_features, factor_file=FACTOR_FILE, lambda_=LAMBDA_REG)
        np.save(NEW_MATRIX_PATH, A)
        with open(CLASS_DICT_PATH, 'w') as f:
            json.dump(class_dict, f, indent=4)
    elif mode == "identify":
        from identify import run_identification
        # Load precomputed factorization (if it exists); else compute full factorization.
        if os.path.exists(FACTOR_FILE):
            L_gallery = torch.load(FACTOR_FILE)
            L_gallery = L_gallery.to(device)
            print("Loaded precomputed gallery factorization.")
        else:
            L_gallery = precompute_gallery_factorization(A, lambda_=LAMBDA_REG)
            torch.save(L_gallery.cpu(), FACTOR_FILE)
            print("Computed and saved gallery factorization.")
        run_identification(A, class_dict, extract_features, L_gallery, lambda_=LAMBDA_REG)

    print("\n--- Operation complete. ---\n")
