import torch
import numpy as np
import os

def identify(y, A, class_dict, lambda_=0.001, device=None):
    """
    Identify the class of a probe vector y given the gallery matrix A and class dictionary.
    This version runs entirely on GPU.
    
    Args:
        y (numpy.ndarray): probe feature vector (1D array)
        A (numpy.ndarray): gallery feature matrix (each column is a feature vector)
        class_dict (dict): mapping from participant IDs to list of column indices in A
        lambda_ (float): regularization parameter
        device (torch.device): device to use (defaults to GPU if available)
        
    Returns:
        best_class (str): the identified class (participant ID) for the probe vector.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Convert gallery matrix and probe vector to GPU tensors
    A_torch = torch.tensor(A, dtype=torch.float32, device=device)
    y_torch = torch.tensor(y, dtype=torch.float32, device=device)
    
    # Compute A^T y and unsqueeze to make it a column vector
    At = A_torch.t()
    Aty = torch.matmul(At, y_torch).unsqueeze(1)  # shape: [n, 1]
    
    # Build the regularized matrix: A^T A + lambda * I
    n = A_torch.shape[1]
    I = torch.eye(n, device=device, dtype=torch.float32)
    regularized_matrix = torch.matmul(At, A_torch) + lambda_ * I

    try:
        # Solve the linear system using Cholesky decomposition on GPU
        L = torch.linalg.cholesky(regularized_matrix)
        x0 = torch.cholesky_solve(Aty, L).squeeze(1)  # x0 shape: [n]
    except RuntimeError as e:
        print("Error during linear solve:", e)
        return None

    best_class = None
    min_error = float('inf')
    # Compute reconstruction error for each class entirely on GPU
    for cls, indices in class_dict.items():
        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        A_class = A_torch[:, indices_tensor]  # shape: (feature_dim, num_class_samples)
        x0_class = x0[indices_tensor]           # shape: (num_class_samples,)
        reconstruction = torch.matmul(A_class, x0_class.unsqueeze(1)).squeeze(1)
        error = torch.norm(y_torch - reconstruction, p=2) ** 2
        if error.item() < min_error:
            min_error = error.item()
            best_class = cls

    return best_class

def run_identification(A, class_dict, extract_features_func, L_gallery, lambda_=0.001):
    """
    Run identification on a set of test images.
    
    Args:
        A (numpy.ndarray): gallery feature matrix
        class_dict (dict): mapping from participant IDs to column indices in A
        extract_features_func (function): function to extract features from an image path
        L_gallery: precomputed gallery factorization (not used in this example but provided for compatibility)
        lambda_ (float): regularization parameter
    """
    test_images_input = input("Enter test image paths (comma-separated): ").strip()
    test_images = [p.strip() for p in test_images_input.split(",") if p.strip()]
    
    if not test_images:
        print("No test image paths provided. Exiting.")
        exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for img_path in test_images:
        feature_vector = extract_features_func(img_path)
        if feature_vector is None:
            print(f"Skipping image {img_path} due to extraction error.")
            continue
        predicted_class = identify(feature_vector, A, class_dict, lambda_=lambda_, device=device)
        print(f"Identified participant for {os.path.basename(img_path)}: {predicted_class}")

if __name__ == "__main__":
    print("This is the identification module.")
