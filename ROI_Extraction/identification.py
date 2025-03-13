import os
import cv2
import json
import numpy as np
import sys
import logging
import argparse
import time
from tqdm import tqdm
import torch
import torchvision.transforms as T
from PIL import Image

LAMBDA_REG = 0.001
# -------------------- DINOv2 Setup -------------------- #

logging.info("Loading DINOv2 model...")
# Loads the DINOv2 model (e.g., "dinov2_vitl14")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logging.info(f"DINOv2 model loaded on device: {device}")

# Define the transformation expected by DINOv2
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# -------------------- Utility Functions -------------------- #

def identify_files(file_path, file_list):
    """
    From a list of base filenames in file_list, find matching ROI image files in file_path.
    """
    file_names = []
    with open(file_list, 'r') as f:
        for line in f:
            parts = line.strip().split('\\')
            if parts:
                file_names.append(parts[-1])
                
    matched_files = []
    for file in os.listdir(file_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Remove "_ROI" from filename to compare
            file_name = os.path.basename(file).replace('_ROI', '')
            if file_name in file_names:
                matched_files.append(os.path.join(file_path, file))
    return matched_files

def load_image_paths(file_path, file_list):
    """
    Return a dictionary mapping participant_number -> list of matched ROI image paths.
    """
    matched_files = identify_files(file_path, file_list)
    classes_with_images = {}
    for fpath in matched_files:
        fname = os.path.basename(fpath)
        participant_number = fname.split('_')[0]
        classes_with_images.setdefault(participant_number, []).append(fpath)
    return classes_with_images

def normalize_vector(vector):
    """
    Normalize a vector to unit norm.
    """
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

def identify(y, A, class_dict, lambda_=0.001):
    """
    Identify the class of a probe vector y given the gallery matrix A and a class dictionary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A_torch = torch.tensor(A, dtype=torch.float32, device=device)
    y_torch = torch.tensor(y, dtype=torch.float32, device=device)
    
    At = A_torch.T
    AtA = At @ A_torch
    Aty = At @ y_torch

    n = A_torch.shape[1]
    I = torch.eye(n, device=device, dtype=torch.float32)
    regularized_matrix = AtA + lambda_ * I

    try:
        x0 = torch.linalg.solve(regularized_matrix, Aty)
    except RuntimeError as e:
        logging.error(f"CUDA error during identification: {e}")
        return None

    x0 = x0.cpu().numpy()

    best_class = None
    min_error = float('inf')
    for cls, indices in class_dict.items():
        A_class = A[:, indices]
        x0_class = x0[indices]
        reconstruction = A_class @ x0_class
        error = np.linalg.norm(y - reconstruction, 2) ** 2
        if error < min_error:
            min_error = error
            best_class = cls

    return best_class

def extract_features_combined(image_path, model, transform):
    """
    Extract a feature vector for the image at image_path using DINOv2.
    """
    try:
        pil_img = Image.open(image_path).convert("L")
        # Replicate grayscale channel to 3-channel
        pil_img = Image.merge("RGB", (pil_img, pil_img, pil_img))
    except Exception as e:
        logging.error(f"Error loading image (PIL): {image_path} : {e}")
        return None

    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        dino_features = model(img_tensor).cpu().numpy().flatten()
    dino_features = normalize_vector(dino_features)
    return dino_features

# -------------------- Main Processing -------------------- #

def main(num_targets, file_path, training_list, testing_list, output_dir):
    print(f"Processing {num_targets} target classes...")
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'identification')
    os.makedirs(output_dir, exist_ok=True)
    # Define paths for files to be saved in the output directory
    final_feature_matrix_path = os.path.join(output_dir, 'feature_matrix_A.npy')
    accuracy_json_file = os.path.join(output_dir, 'accuracy.json')
    misclassified_txt_file = os.path.join(output_dir, f'misclassified_for_{num_targets}classes.txt')

    # Generate target classes (e.g. '001', '002', ..., based on num_targets)
    target_classes = [f"{i:03d}" for i in range(1, num_targets + 1)]
    logging.info(f"Target classes: {target_classes}")
    
    # Load training image paths
    known_classes_with_images = load_image_paths(file_path, training_list)
    logging.info(f"Loaded image paths for {len(known_classes_with_images)} classes (training set).")

    # Gather training samples for target classes
    training_samples = []
    for cls_label, image_paths in known_classes_with_images.items():
        if cls_label in target_classes:
            for img_path in image_paths:
                training_samples.append((cls_label, img_path))
    total_samples = len(training_samples)
    logging.info(f"Total training samples to process: {total_samples}")

    if total_samples == 0:
        logging.error("No training samples found for the specified target classes.")
        sys.exit(1)

    # Determine feature dimension from first valid training sample
    feature_dim = None
    for cls_label, img_path in training_samples:
        feature = extract_features_combined(img_path, model, transform)
        if feature is not None:
            feature_dim = feature.shape[0]
            break
    if feature_dim is None:
        logging.error("Failed to determine feature dimension from training samples.")
        sys.exit(1)
    logging.info(f"Determined feature dimension: {feature_dim}")

    # ------------------ Feature Extraction and Matrix Building ------------------ #
    extraction_time_total = 0.0
    matrix_build_time = 0.0
    class_dict = {}

    if num_targets == 200:
        t_memmap_start = time.time()
        A_memmap = np.memmap(final_feature_matrix_path, dtype='float32', mode='w+', shape=(feature_dim, total_samples))
        current_index = 0
        for cls_label, img_path in tqdm(training_samples, desc="Processing training images (memmap)"):
            t_extraction = time.time()
            features = extract_features_combined(img_path, model, transform)
            t_extraction_end = time.time()
            extraction_time_total += (t_extraction_end - t_extraction)
            if features is None:
                continue
            A_memmap[:, current_index] = features
            class_dict.setdefault(cls_label, []).append(current_index)
            current_index += 1
        t_memmap_end = time.time()
        # Matrix building time is the remainder of the memmap loop time
        matrix_build_time = (t_memmap_end - t_memmap_start) - extraction_time_total

        if current_index < total_samples:
            logging.info(f"Processed {current_index} samples out of {total_samples}. Adjusting feature matrix.")
            A_final = np.array(A_memmap[:, :current_index])
        else:
            A_final = np.array(A_memmap)
        
        A_memmap.flush()
        del A_memmap
        np.save(final_feature_matrix_path, A_final)
        A = np.load(final_feature_matrix_path, allow_pickle=True)
        logging.info(f"Final feature matrix shape (stored to disk): {A.shape}")
    else:
        t_list_start = time.time()
        features_list = []
        current_index = 0
        for cls_label, img_path in tqdm(training_samples, desc="Processing training images (in memory)"):
            t_extraction = time.time()
            features = extract_features_combined(img_path, model, transform)
            t_extraction_end = time.time()
            extraction_time_total += (t_extraction_end - t_extraction)
            if features is None:
                continue
            features_list.append(features)
            class_dict.setdefault(cls_label, []).append(current_index)
            current_index += 1
        t_list_end = time.time()
        matrix_build_time = (t_list_end - t_list_start) - extraction_time_total
        if len(features_list) == 0:
            logging.error("No valid training samples processed.")
            sys.exit(1)
        A = np.column_stack(features_list)
        logging.info(f"Final feature matrix shape (in memory): {A.shape}")
    
    logging.info(f"Class dictionary (sample counts): { {k: len(v) for k, v in class_dict.items()} }")
    logging.info(f"Total feature extraction time: {extraction_time_total:.2f} sec")
    logging.info(f"Total matrix building time: {matrix_build_time:.2f} sec")
    print(f"Feature extraction time: {extraction_time_total:.2f} sec")
    print(f"Matrix building time: {matrix_build_time:.2f} sec")

    # -------------------- Phase 2: Identification (Classification) -------------------- #

    test_files = identify_files(file_path, testing_list)
    valid_files = []
    for path in test_files:
        participant_number = os.path.basename(path).split('_')[0]
        if participant_number in target_classes:
            valid_files.append(path)
    print(f"Found {len(valid_files)} test files in {file_path} (from {testing_list}) matching target classes.")
    logging.info(f"Chose {len(valid_files)} images for classification.")

    # Prepare lists for metrics and misclassified images
    actual_labels = []
    predicted_labels = []
    misclassified_list = []

    # Start timing classification phase
    start_time = time.time()

    def classify_image(path, A, class_dict, model, transform, lambda_=0.001):
        try:
            features = extract_features_combined(path, model, transform)
            if features is None:
                return None
            identified_class = identify(features, A, class_dict, lambda_=lambda_)
            participant = os.path.basename(path).split('_')[0]
            logging.info(f"File: {os.path.basename(path)}, Actual: {participant}, Predicted: {identified_class}")
            print(f"Actual: {participant}, Predicted: {identified_class}")
            return identified_class
        except Exception as e:
            logging.error(f"Error classifying {path}: {e}")
            print(f"Error classifying {path}: {e}")
            return None

    correct_count = 0
    incorrect_count = 0
    for img_path in tqdm(valid_files, desc="Classifying test images"):
        predicted = classify_image(img_path, A, class_dict, model, transform, lambda_=LAMBDA_REG)
        if predicted is not None:
            actual = os.path.basename(img_path).split('_')[0]
            actual_labels.append(actual)
            predicted_labels.append(predicted)
            if predicted == actual:
                correct_count += 1
            else:
                incorrect_count += 1
                misclassified_list.append(img_path)

    end_time = time.time()
    classification_time = end_time - start_time
    total_classified = len(actual_labels)
    overall_accuracy = (correct_count / total_classified * 100) if total_classified else 0.0
    processing_speed =   total_classified/classification_time if total_classified > 0 else None

    print("\n--- Classification Summary ---")
    print(f"Correct predictions:   {correct_count}")
    print(f"Incorrect predictions: {incorrect_count}")
    print(f"Overall Accuracy:      {overall_accuracy:.2f}%")
    print(f"Classification time:   {classification_time:.2f} seconds")
    print(f"Processing speed:      {processing_speed:.2f} images/sec")
    logging.info(f"Correct predictions: {correct_count}, Incorrect predictions: {incorrect_count}, Overall Accuracy: {overall_accuracy:.2f}%")
    logging.info(f"Classification time: {classification_time:.2f} sec, Processing speed: {processing_speed:.2f} images/sec")
    print("Classification phase completed.")

  

    # -------------------- Compute Additional Metrics -------------------- #

    # Build confusion matrix (dictionary of dictionaries)
    confusion_matrix = {cls: {cls2: 0 for cls2 in target_classes} for cls in target_classes}
    for a, p in zip(actual_labels, predicted_labels):
        if a in target_classes and p in target_classes:
            confusion_matrix[a][p] += 1

    # Compute per-class accuracy, precision, recall, and F1-score
    per_class_accuracy = {}
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}

    for cls in target_classes:
        TP = confusion_matrix[cls][cls]
        total_actual = sum(confusion_matrix[cls].values())
        per_class_accuracy[cls] = (TP / total_actual * 100) if total_actual > 0 else None

        total_predicted = sum(confusion_matrix[r][cls] for r in target_classes)
        precision_dict[cls] = (TP / total_predicted * 100) if total_predicted > 0 else None
        recall_dict[cls] = (TP / total_actual * 100) if total_actual > 0 else None

        if precision_dict[cls] is not None and recall_dict[cls] is not None and (precision_dict[cls] + recall_dict[cls]) > 0:
            f1_dict[cls] = 2 * (precision_dict[cls] * recall_dict[cls]) / (precision_dict[cls] + recall_dict[cls])
        else:
            f1_dict[cls] = None

    # Compute overall (macro-averaged) precision, recall, and F1-score
    valid_precisions = [precision_dict[cls] for cls in target_classes if precision_dict[cls] is not None]
    overall_precision = sum(valid_precisions) / len(valid_precisions) if valid_precisions else None

    valid_recalls = [recall_dict[cls] for cls in target_classes if recall_dict[cls] is not None]
    overall_recall = sum(valid_recalls) / len(valid_recalls) if valid_recalls else None

    valid_f1s = [f1_dict[cls] for cls in target_classes if f1_dict[cls] is not None]
    overall_f1 = sum(valid_f1s) / len(valid_f1s) if valid_f1s else None

    # -------------------- Append All Metrics to JSON -------------------- #
    record = {
    num_targets: {
        "method": "DINOv2_vitl14",
        "timing": {
            "feature_extraction_sec": extraction_time_total,
            "matrix_build_sec": matrix_build_time,
            "classification_sec": classification_time,
            "classification_speed_img_per_sec": processing_speed
        },
        "results": {
            "accuracy": {
                "overall_percent": overall_accuracy,
                "per_class_percent": per_class_accuracy
            },
            "confusion_matrix": confusion_matrix,
            "performance": {
                "precision": {
                    "per_class_percent": precision_dict,
                    "macro_percent": overall_precision
                },
                "recall": {
                    "per_class_percent": recall_dict,
                    "macro_percent": overall_recall
                },
                "f1_score": {
                    "per_class": f1_dict,
                    "macro_percent": overall_f1
                }
            }
        },
        "misclassified_images": misclassified_list,
        "metadata":{
            "num_training_samples": total_samples,
            "num_test_samples": len(valid_files),
            "correct_predictions": correct_count,
            "feature_dim": feature_dim,
        }  
    }
}

    
    # Load existing accuracy JSON data (structured as a list)
    if os.path.exists(accuracy_json_file):
        with open(accuracy_json_file, 'r') as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError:
                json_data = []
    else:
        json_data = []
    
    json_data.append(record)
    
    with open(accuracy_json_file, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    logging.info("Metrics record appended to %s", accuracy_json_file)
    print(f"Metrics record appended to {accuracy_json_file}")


