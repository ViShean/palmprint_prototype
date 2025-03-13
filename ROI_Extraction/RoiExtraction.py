# ROI Extraction.py 
import itertools
import os
import re
import subprocess

import cv2
import numpy as np

def run_darknet_detection(
    darknet_dir,
    obj_data_path,
    cfg_file,
    weights_file,
    input_image,
    output_dir,
    threshold=0.5,
    iou_threshold=0.3,  # Adjust as needed
    max_tries=10,
    min_threshold=0.1,
    max_threshold=0.9
):
    """
    Runs Darknet object detection on a single image with NMS to ensure 4 distinct boxes.

    Parameters:
    - darknet_dir (str): Path to the Darknet directory containing darknet.exe
    - obj_data_path (str): Path to the obj.data file
    - cfg_file (str): Path to the cfg file
    - weights_file (str): Path to the weights file
    - input_image (str): Path to the input image
    - output_dir (str): Directory to save the output
    - threshold (float): Starting detection threshold
    - iou_threshold (float): IoU threshold for NMS
    - max_tries (int): Maximum number of threshold adjustments
    - min_threshold (float): Minimum threshold value
    - max_threshold (float): Maximum threshold value

    Returns:
    - list of dicts: Filtered detections with exactly four distinct, non-overlapping boxes
    """
    os.makedirs(output_dir, exist_ok=True)
    # darknet_exe = os.path.join(darknet_dir, "darknet.exe")
    
    if os.name == 'nt':  # Windows
        darknet_exe = os.path.join(darknet_dir, "darknet.exe")
    else:  # Unix-based systems (Linux/Mac)
        darknet_exe = os.path.join(darknet_dir, "darknet")  # Pointing to the correct location

    if not os.path.isfile(darknet_exe):
        print(f"Error: 'darknet.exe' not found in directory: {darknet_dir}")
        return []

    tries = 0

    while tries < max_tries:
        tries += 1
        print(f"\nAttempt {tries}: Running detection with threshold {threshold}")
        command = [
            darknet_exe,
            "detector",
            "test",
            obj_data_path,
            cfg_file,
            weights_file,
            input_image,
            "-thresh",
            str(threshold),
            "-dont_show",
            "-ext_output"
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=darknet_dir  # Ensure the working directory is set
            )
            raw_output = result.stdout
            detections = parse_darknet_output(raw_output)
            detected_points = len(detections)
            print(f"Detected points before NMS: {detected_points}")

            if detected_points == 0:
                print(f"No detections found for {input_image} with threshold {threshold}.")
                threshold -= 0.1  # Decrease threshold to allow more detections
                if threshold < min_threshold:
                    print(f"Threshold below minimum limit ({min_threshold}). Stopping detection.")
                    break
                continue

            # Prepare data for NMS
            boxes = []
            confidences = []
            for det in detections:
                x, y, w, h = det['bbox']
                boxes.append([int(x), int(y), int(w), int(h)])
                confidences.append(float(det['confidence']))

            # Apply Non-Maximum Suppression (NMS)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=threshold, nms_threshold=iou_threshold)

            if len(indices) == 0:
                print(f"All detections suppressed after NMS for {input_image}. Adjusting threshold.")
                threshold -= 0.05  # Adjust threshold to allow more detections
                if threshold < min_threshold:
                    print(f"Threshold below minimum limit ({min_threshold}). Stopping detection.")
                    break
                continue

            # Extract NMS-filtered boxes and corresponding detections
            nms_boxes = []
            nms_confidences = []
            nms_detections = []

            for i in indices:
                if isinstance(i, (list, tuple, np.ndarray)):
                    idx = i[0]
                else:
                    idx = i
                nms_boxes.append(boxes[idx])
                nms_confidences.append(confidences[idx])
                nms_detections.append(detections[idx])

            print(f"NMS filtered boxes count: {len(nms_boxes)}")

            # If more than 4 boxes, select top 4 based on confidence
            if len(nms_boxes) > 4:
                sorted_indices = np.argsort(nms_confidences)[::-1]  # Descending order
                nms_boxes = [nms_boxes[i] for i in sorted_indices[:4]]
                nms_confidences = [nms_confidences[i] for i in sorted_indices[:4]]
                nms_detections = [nms_detections[i] for i in sorted_indices[:4]]
                print("Selected top 4 boxes based on confidence scores.")

            # Check if exactly 4 boxes are present
            if len(nms_boxes) == 4:
                print(f"Successfully detected 4 boxes on attempt {tries} with threshold {threshold}")
                # Reconstruct detections with NMS results
                final_detections = []
                for det in nms_detections:
                    detection = {
                        'class_name': det['class_name'],  # Use actual class name
                        'confidence': det['confidence'],
                        'bbox': det['bbox'],
                        'coordinate': det['coordinate']
                    }
                    final_detections.append(detection)
                return final_detections
            elif len(nms_boxes) < 4:
                if threshold > min_threshold:
                    print(f"Detected points ({len(nms_boxes)}) less than 4. Decreasing threshold.")
                    threshold -= 0.05  # Decrease threshold to allow more detections
                else:
                    print(f"Threshold reached minimum limit ({min_threshold}). Cannot detect 4 boxes.")
                    break
            else:
                # More than 4 boxes but already handled by selecting top 4
                print(f"Tried: {tries}, Threshold: {threshold}")
                final_detections = []
                for det in nms_detections:
                    detection = {
                        'class_name': det['class_name'],  # Use actual class name
                        'confidence': det['confidence'],
                        'bbox': det['bbox'],
                        'coordinate': det['coordinate']
                    }
                    final_detections.append(detection)
                return final_detections

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the detection command for {input_image}:")
            print(e.stderr)
            return []

    print(f"Exceeded maximum tries for {input_image}. Returning available detections.")
    return detections

def parse_darknet_output(darknet_output):
    """
    Parses the Darknet detection output.
    """
    detections = []
    detection_pattern = re.compile(
        r'^(?P<class_name>[^:]+):\s+(?P<confidence>\d+)%\s+\(left_x:\s*(?P<left_x>\d+)\s+top_y:\s*(?P<top_y>\d+)\s+width:\s*(?P<width>\d+)\s+height:\s*(?P<height>\d+)\)$'
    )

    for line in darknet_output.split('\n'):
        match = detection_pattern.search(line)
        if match:
            class_name = match.group(1)
            confidence = float(match.group(2)) / 100.0
            left_x = int(match.group(3))
            top_y = int(match.group(4))
            width = int(match.group(5))
            height = int(match.group(6))
            x1 = left_x
            y1 = top_y
            x2 = left_x + width
            y2 = top_y + height
            # Calculate the center point of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            detections.append({
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [left_x, top_y, width, height],
                'coordinate': [center_x, center_y]
            })
    return detections

def find_closest_trio(points):
    """
    Finds the trio of points that are closest to each other.

    Returns:
    - tuple: (list of tuples) The three closest points, (tuple) the fourth point.
    """
    if len(points) < 4:
        print("Not enough points to find a trio and a fourth point.")
        return [], None

    min_total_distance = float('inf')
    closest_trio = []
    thumb_index_gap = None

    # Iterate over all possible combinations of three points
    for trio in itertools.combinations(points, 3):
        # Calculate pairwise distances
        d1 = np.linalg.norm(np.array(trio[0]) - np.array(trio[1]))
        d2 = np.linalg.norm(np.array(trio[0]) - np.array(trio[2]))
        d3 = np.linalg.norm(np.array(trio[1]) - np.array(trio[2]))
        total_distance = d1 + d2 + d3

        if total_distance < min_total_distance:
            min_total_distance = total_distance
            closest_trio = trio
            # Find the fourth point not in the trio
            thumb_index_gap = next((p for p in points if p not in trio), None)
    
    return closest_trio, thumb_index_gap

def extract_points_from_detections(detections):
    """
    Extracts center points (x, y) from the detection bounding boxes.

    Parameters:
    - detections (list of dicts): Detection results, each containing a 'bbox' key.

    Returns:
    - list of tuples: A list of (x, y) coordinates representing center points.
    """
    points = []
    for det in detections:
        # Extract bounding box
        left_x, top_y, width, height = det['bbox']
        # Calculate center coordinates
        center_x = int(left_x + width / 2)
        center_y = int(top_y + height / 2)
        # Append center point as a tuple
        points.append((center_x, center_y))
    return points

def calculate_midpoints(trio, thumb_index_gap):
    """
    Calculates the midpoints between P1-P2 and P2-P3, ensuring the points are ordered
    based on their orientation with respect to the thumb-index gap point.

    Parameters:
    - trio (list of tuples): The three closest points as (x, y) tuples.
    - thumb_index_gap (tuple): Coordinates of the thumb-index gap point.

    Returns:
    - tuple: Midpoints between P1-P2 and P2-P3 as ((x_mid1, y_mid1), (x_mid2, y_mid2)).
    """
    if len(trio) != 3:
        raise ValueError("Trio must contain exactly three points.")

    # Convert points to numpy arrays
    trio = [np.array(point) for point in trio]
    thumb_index_gap = np.array(thumb_index_gap)

    # Calculate centroid of the trio
    centroid = sum(trio) / 3

    # Calculate vectors from centroid to trio points
    vectors = [point - centroid for point in trio]

    # Calculate vector from centroid to thumb-index gap point
    vector_to_thumb = thumb_index_gap - centroid

    # Calculate angles between vectors to points and vector to thumb-index gap
    angles = []
    for point, vector in zip(trio, vectors):
        # Use the cross product to determine relative orientation
        cross_product = np.cross(vector, vector_to_thumb)
        angles.append((point, cross_product))

    # Sort the trio points based on the cross product values
    # This will order the points consistently relative to the thumb-index gap point
    sorted_trio = [point for point, _ in sorted(angles, key=lambda x: x[1])]

    # Extract ordered points
    P1, P2, P3 = sorted_trio

    # Calculate midpoints
    midpoint1 = ((P1[0] + P2[0]) / 2, (P1[1] + P2[1]) / 2)
    midpoint2 = ((P2[0] + P3[0]) / 2, (P2[1] + P3[1]) / 2)

    return midpoint1, midpoint2

def calculate_point_c(midpoint1, midpoint2, thumb_index_gap):
    """
    Calculates Point C based on midpoints A and B, ensuring its orientation
    is directed appropriately based on the thumb-index gap point.

    Parameters:
    - midpoint1 (tuple): Midpoint between P1-P2.
    - midpoint2 (tuple): Midpoint between P2-P3.
    - thumb_index_gap (tuple): Coordinates of the thumb-index gap point.

    Returns:
    - tuple: Coordinates of Point C.
    """
    # Convert points to numpy arrays
    midpoint1 = np.array(midpoint1)
    midpoint2 = np.array(midpoint2)
    O = (midpoint1 + midpoint2) / 2
    thumb_index_gap = np.array(thumb_index_gap)

    # Vector from midpoint1 to midpoint2 (AB)
    AB = midpoint2 - midpoint1
    AB_length = np.linalg.norm(AB)

    if AB_length == 0:
        raise ValueError("AB length is zero, cannot determine perpendicular direction.")

    # Unit vector along AB
    AB_unit = AB / AB_length

    # Perpendicular vector to AB
    perp_vector = np.array([-AB_unit[1], AB_unit[0]])

    # Vector from O to thumb-index gap point
    vector_to_thumb = thumb_index_gap - O

    # Determine on which side of AB the thumb-index gap point lies
    # Using the sign of the cross product
    cross_product = np.cross(AB_unit, vector_to_thumb)

    # If cross_product is negative, thumb-index gap is on one side; if positive, on the other
    # Adjust the direction of perp_vector accordingly
    if cross_product < 0:
        perp_vector = -perp_vector

    # Calculate Point C
    OC_length = 1.5 * AB_length  # Adjust the multiplier as needed
    C = O + perp_vector * OC_length
    C = tuple(map(int, C))

    return C

def extract_roi(image, midpoints, point_c, thumb_index_gap,hand_type='right'):
    """
    Extracts the ROI from the image based on the midpoints and Point C,
    ensuring the ROI is parallel to the gradient between the midpoints and
    oriented towards the thumb-index gap.

    Parameters:
    - image (numpy.ndarray): The original image.
    - midpoints (tuple): Midpoints A and B.
    - point_c (tuple): Coordinates of Point C (center of the ROI).
    - thumb_index_gap (tuple): Coordinates of the thumb-index gap point.

    Returns:
    - numpy.ndarray: The cropped ROI image.
    """
    # Calculate vector from midpoint1 to midpoint2
    midpoint1, midpoint2 = midpoints
    vector_midpoints = np.array(midpoint2) - np.array(midpoint1)
    length_midpoints = np.linalg.norm(vector_midpoints)

    # Calculate angle of rotation in degrees
    angle = np.degrees(np.arctan2(vector_midpoints[1], vector_midpoints[0]))

    # Determine the orientation based on the thumb-index gap
    # Calculate vector from Point C to thumb-index gap
    vector_to_thumb = np.array(thumb_index_gap) - np.array(point_c)
    dot_product = np.dot(vector_midpoints, vector_to_thumb)
    if hand_type == 'right':
        if dot_product < 0:
            angle += 180
    else:  # hand_type == 'left'
        # Invert the logic for left hand
        if dot_product > 0:
            angle += 180

    # Define the size of the ROI
    # Adjust the width and height based on your requirements
    side_length = length_midpoints * 2.5
    width = side_length
    height = side_length


    # Create a RotatedRect object
    rect = ((point_c[0], point_c[1]), (width, height), angle)

    # Get the rotation matrix for the affine transform
    M = cv2.getRotationMatrix2D((point_c[0], point_c[1]), angle, 1.0)

    # Perform the affine transformation (rotate the entire image)
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Extract the ROI from the rotated image
    # Compute the bounding box of the rotated rectangle
    roi = cv2.getRectSubPix(
        rotated_image,
        (int(width), int(height)),
        (point_c[0], point_c[1])
    )

    # Compute the corners of the rotated rectangle (for annotation)
    box = cv2.boxPoints(rect)
    box = np.intp(box)  # Convert to integer coordinates

    return roi, box


