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
def run_identification(A, class_dict, extract_features_func, img_path):
    """
    Identify participant for a single test image using nearest neighbor search.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_torch = torch.tensor(A, dtype=torch.float32, device=device)

    inv_dict = {}
    for cls, indices in class_dict.items():
        for idx in indices:
            inv_dict[idx] = cls

    feature_vector = extract_features_func(img_path)
    if feature_vector is None:
        print(f"Skipping image {img_path} due to extraction error.")
        return None

    q = torch.tensor(feature_vector, dtype=torch.float32, device=device)
    similarity = torch.matmul(q.unsqueeze(0), A_torch)
    nn_index = torch.argmax(similarity).item()
    predicted_class = inv_dict.get(nn_index, None)
    print(f"Identified participant for {os.path.basename(img_path)}: {predicted_class}")
    return predicted_class


if __name__ == "__main__":
    print("This is the brute-force identification module.")
