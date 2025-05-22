import os
import numpy as np
import json

def run_enrollment(A, class_dict, new_matrix_path, extract_features_func):
    """
    Enroll new participant images by extracting features, updating the gallery matrix A,
    and updating the class dictionary.
    
    Returns:
      Updated A (gallery matrix) and updated class_dict.
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
    
    new_features = np.array(new_features_list)  # shape: (num_images, d)
    print(f"New enrollment features shape: {new_features.shape}")
    
    # Check that the feature dimension is consistent.
    if new_features.shape[1] != A.shape[0]:
        print(f"Feature dimension mismatch: enrollment features have dimension {new_features.shape[1]} but gallery features have dimension {A.shape[0]}. Exiting.")
        exit(1)
    
    # Transpose new_features so that each column is a feature vector.
    new_features_T = new_features.T  # shape: (d, num_images)
    print(f"Original feature matrix shape: {A.shape}")
    A_expanded = np.hstack((A, new_features_T))
    np.save(new_matrix_path, A_expanded)
    print(f"Updated feature matrix shape: {A_expanded.shape}")
    
    # Update class dictionary with the indices for the new participant.
    start_index = A.shape[1]
    new_indices = list(range(start_index, start_index + new_features_T.shape[1]))
    class_dict[new_participant_id] = new_indices
    print(f"New participant {new_participant_id} enrolled successfully!")
    
    return A_expanded, class_dict
