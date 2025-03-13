import os
import numpy as np
import json
import torch

def incremental_cholesky_update(L_old, A_old, x, lambda_):
    """
    Incrementally update the Cholesky factorization when a new column x is added.
    Given:
      L_old: current lower-triangular Cholesky factor for M_old = A_old^T A_old + lambda I (shape: [n, n])
      A_old: current gallery matrix as a NumPy array (shape: [m, n])
      x: new column vector (NumPy array of shape: [m,])
      lambda_: regularization parameter.
    Returns:
      L_new: updated Cholesky factor for the new matrix M_new = A_new^T A_new + lambda I,
             where A_new = [A_old, x] (shape: [n+1, n+1])
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert A_old and x to tensors on device
    A_old_t = torch.tensor(A_old, dtype=torch.float32, device=device)
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    # Compute b = A_old^T x, shape: [n]
    b = torch.matmul(A_old_t.t(), x_tensor)
    # Solve L_old * w = b for w
    w = torch.linalg.solve(L_old, b)
    # Compute d = x^T x + lambda - w^T w
    d = torch.dot(x_tensor, x_tensor) + lambda_ - torch.dot(w, w)
    if d <= 0:
        raise ValueError("Non-positive value encountered in incremental update.")
    alpha = torch.sqrt(d)
    n = L_old.shape[0]
    # Construct L_new as a block matrix:
    # L_new = [ L_old      0 ]
    #         [ w^T      alpha ]
    L_new = torch.zeros((n+1, n+1), device=device, dtype=torch.float32)
    L_new[:n, :n] = L_old
    L_new[n, :n] = w
    L_new[n, n] = alpha
    return L_new

def run_enrollment(A, class_dict, new_matrix_path, extract_features_func, factor_file=None, lambda_=0.001):
    """
    Enroll new participant images, update the gallery matrix, update the class dictionary,
    and incrementally update the precomputed gallery factorization.
    If factor_file is provided and exists, it is loaded and updated; otherwise, factorization is computed from scratch.
    """
    new_participant_id = input("Enter new participant ID: ").strip()
    new_images_input = input("Enter enrollment image paths (comma-separated): ").strip()
    new_images = [p.strip() for p in new_images_input.split(",") if p.strip()]
    
    if not new_images:
        print("No enrollment image paths provided. Exiting.")
        exit(1)
    
    new_features_list = []
    for img_path in new_images:
        feature_vector = extract_features_func(img_path)
        if feature_vector is not None:
            new_features_list.append(feature_vector)
    
    if not new_features_list:
        print("No valid images found for new participant. Exiting.")
        exit(1)
    
    new_features = np.array(new_features_list)  # shape: (num_images, feature_dim)
    print(f"New enrollment features shape: {new_features.shape}")
    
    if new_features.shape[1] != A.shape[0]:
        print(f"Feature dimension mismatch: enrollment features have dimension {new_features.shape[1]} but gallery features have dimension {A.shape[0]}. Exiting.")
        exit(1)
    
    # Transpose new_features so each column is a feature vector
    new_features_T = new_features.T  # shape: (feature_dim, num_new_samples)
    print(f"Original feature matrix shape: {A.shape}")
    A_expanded = np.hstack((A, new_features_T))
    np.save(new_matrix_path, A_expanded)
    print(f"Updated feature matrix shape: {A_expanded.shape}")
    
    start_index = A.shape[1]
    new_indices = list(range(start_index, start_index + new_features_T.shape[1]))
    class_dict[new_participant_id] = new_indices
    print(f"New participant {new_participant_id} enrolled successfully!")
    
    # Incrementally update the gallery factorization.
    # First, load existing factorization if available; else, compute on current A_old (gallery before new columns).
    if factor_file and os.path.exists(factor_file):
        L = torch.load(factor_file, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Loaded existing gallery factorization.")
    else:
        # Compute from scratch on the old gallery matrix
        A_old = A_expanded[:, :start_index]  # gallery before new columns
        A_old_t = torch.tensor(A_old, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        At = A_old_t.t()
        AtA = torch.matmul(At, A_old_t)
        n = A_old_t.shape[1]
        I = torch.eye(n, device=A_old_t.device, dtype=torch.float32)
        regularized_matrix = AtA + lambda_ * I
        L = torch.linalg.cholesky(regularized_matrix)
        print("Computed gallery factorization from scratch.")
    
    # Now, update L for each new column in new_features_T
    A_old = A_expanded[:, :start_index]  # current old gallery before new enrollments
    for j in range(new_features_T.shape[1]):
        x = new_features_T[:, j]  # new feature column (1D NumPy array)
        # Update L incrementally
        L = incremental_cholesky_update(L, A_old, x, lambda_=lambda_)
        # Append x to A_old (convert x to column vector) for next update
        A_old = np.hstack((A_old, x.reshape(-1, 1)))
    
    # Save updated factorization to disk (store on CPU for portability)
    if factor_file:
        torch.save(L.cpu(), factor_file)
        print(f"Updated gallery factorization saved to {factor_file}.")
    
    return A_expanded, class_dict
