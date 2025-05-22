import torch
import numpy as np
import os

def invert_class_dict(class_dict):
    """
    Invert the class dictionary mapping from class -> list of gallery indices
    to a dictionary mapping gallery index -> class.
    """
    inv = {}
    for cls, indices in class_dict.items():
        for idx in indices:
            inv[idx] = cls
    return inv

def run_identification(A, class_dict, extract_features_func):
    """
    Identify participants for test images using brute-force nearest neighbor search.
    
    For each test image:
      - Extract features.
      - Transfer the probe vector and gallery matrix to GPU.
      - Compute inner product similarities.
      - Find the index of the highest similarity.
      - Map the index to the corresponding participant using the inverted dictionary.
    """
    test_images_input = input("Enter test image paths (comma-separated): ").strip()
    test_images = [p.strip() for p in test_images_input.split(",") if p.strip()]
    if not test_images:
        print("No test image paths provided. Exiting.")
        exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Transfer gallery matrix A to GPU once.
    A_torch = torch.tensor(A, dtype=torch.float32, device=device)
    # Invert the class dictionary to map gallery index -> class.
    inv_dict = {}
    for cls, indices in class_dict.items():
        for idx in indices:
            inv_dict[idx] = cls
    
    for img_path in test_images:
        feature_vector = extract_features_func(img_path)
        if feature_vector is None:
            print(f"Skipping image {img_path} due to extraction error.")
            continue
        # Convert probe feature vector to a GPU tensor.
        q = torch.tensor(feature_vector, dtype=torch.float32, device=device)  # shape: (d,)
        # Compute similarities between q and every column in A.
        similarity = torch.matmul(q.unsqueeze(0), A_torch)  # shape: (1, n)
        nn_index = torch.argmax(similarity).item()
        predicted_class = inv_dict.get(nn_index, None)
        print(f"Identified participant for {os.path.basename(img_path)}: {predicted_class}")

if __name__ == "__main__":
    print("This is the brute-force identification module.")
