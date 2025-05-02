# enroll.py
"""
Single-image enrolment helper for the palm-print pipeline.
Compatible with Python 3.6.
"""

import numpy as np
import logging
import pathlib


def run_enrollment(A,
                   class_dict,
                   new_matrix_path: pathlib.Path,
                   extract_features_func,
                   img_path: str,
                   participant_id: str = None):
    """
    Add ONE palm image to the gallery matrix.

    Parameters
    ----------
    A : np.ndarray
        Current gallery of shape (d, n).
    class_dict : dict
        Maps participant_id -> list[column_indices].
    new_matrix_path : pathlib.Path
        Where to save the expanded matrix (.npy).
    extract_features_func : callable
        Function that returns a 1-D feature vector for an image path.
    img_path : str
        Path to the enrolment image.
    participant_id : str, optional
        If supplied and already present in class_dict, append to that
        participant; otherwise a new numeric ID is allocated.

    Returns
    -------
    A_expanded : np.ndarray
    class_dict : dict
    used_id    : str   The participant ID that received this image.
    """

    # ------------------------------------------------------ choose ID
    if participant_id and participant_id in class_dict:
        used_id = participant_id
    else:
        existing_numeric = [int(k) for k in class_dict.keys() if k.isdigit()]
        next_id = max(existing_numeric) + 1 if existing_numeric else 1
        used_id = str(next_id)

    # ------------------------------------------------- feature vector
    feat = extract_features_func(img_path)
    if feat is None:
        logging.error("Feature extraction failed for %s", img_path)
        return A, class_dict, None

    feat = feat.astype(np.float32).reshape(1, -1)       # (1, d)

    # ------------------------------------------------ dim check
    if feat.shape[1] != A.shape[0]:
        logging.error("Dim mismatch: feat %d vs gallery %d",
                      feat.shape[1], A.shape[0])
        return A, class_dict, None

    # --------------------------------------------- append and save
    feat_T = feat.T                                     # (d, 1)
    A_expanded = np.hstack((A, feat_T))
    np.save(str(new_matrix_path), A_expanded)

    # --------------------------------------------- update mapping
    new_col = A.shape[1]                                # index of new column
    class_dict.setdefault(used_id, []).append(new_col)

    logging.info("Enrolled img %s as participant %s (col %d)",
                 img_path, used_id, new_col)
    return A_expanded, class_dict, used_id
