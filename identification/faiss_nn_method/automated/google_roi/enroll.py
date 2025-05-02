import os
import numpy as np
import json

def run_enrollment(A, class_dict, new_matrix_path, extract_features_func, img_path):
    """
    Enrol a single image:
      • extract its feature vector
      • append it as one new column to A
      • add a new participant ID in class_dict
    Returns the updated (A, class_dict).
    """

    # -------- auto-assign next numeric ID -------------------------------
    existing = [int(k) for k in class_dict.keys() if k.isdigit()]
    next_id  = max(existing) + 1 if existing else 1
    new_participant_id = str(next_id)

    # -------- feature extraction ----------------------------------------
    feature_vector = extract_features_func(img_path)

    if feature_vector is None:
        print("Feature extraction failed for", img_path)
        return A, class_dict

    # ---- wrap single vector into 2-D array (num_images=1) --------------  # ← changed
    new_features = np.expand_dims(feature_vector, axis=0)   # shape (1, d)  # ← changed
    print("New enrollment features shape:", new_features.shape)

    # ---- dimension check ------------------------------------------------
    if new_features.shape[1] != A.shape[0]:
        print("Feature dimension mismatch — exiting.")
        return A, class_dict

    # ---- append column --------------------------------------------------
    new_features_T = new_features.T                     # shape (d, 1)
    A_expanded = np.hstack((A, new_features_T))
    np.save(new_matrix_path, A_expanded)
    print("Updated feature matrix shape:", A_expanded.shape)

    # ---- update class dictionary ---------------------------------------
    new_index = A.shape[1]                              # first new column
    class_dict[new_participant_id] = [new_index]        # ← changed
    print("New participant", new_participant_id, "enrolled successfully!")

    return A_expanded, class_dict
