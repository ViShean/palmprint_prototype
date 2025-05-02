# google_roi.py
"""
Hand-ROI extractor based on MediaPipe Hands (single hand, 21 landmarks).

Public API
----------
extract_palm_roi(image: numpy.ndarray) -> (roi_bgr, annotated_bgr, hand_type)

• roi_bgr          – cropped palm ROI (BGR)
• annotated_bgr    – same size as input with landmarks & box drawn
• hand_type        – "Left" / "Right" / "Unknown"
"""

import cv2
import numpy as np
import mediapipe as mp

# ----------------------------------------------------------------------
# Internal helpers (mostly identical to your original script)
# ----------------------------------------------------------------------

def _run_mp_hands(image, min_det_conf=0.2):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_det_conf
    ) as hands:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, None
        lm = res.multi_hand_landmarks[0]
        handedness = (
            res.multi_handedness[0].classification[0].label
            if res.multi_handedness else "Unknown"
        )
        h, w = image.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
        return pts, handedness

def _midpoints(pairs):
    return [((p1[0]+p2[0])/2, (p1[1]+p2[1])/2) for p1, p2 in pairs]

def _calculate_point_c(m1, m2, thumb):
    m1, m2, thumb = map(np.asarray, (m1, m2, thumb))
    O  = (m1 + m2) / 2.0
    AB = m2 - m1
    L  = np.linalg.norm(AB)
    if L == 0: raise ValueError("Midpoints coincide")
    ABu = AB / L
    perp = np.array([-ABu[1], ABu[0]])
    if np.cross(ABu, thumb - O) < 0:
        perp = -perp
    return tuple((O + 1.8 * L * perp).astype(int))  # 1.8 == scale factor

def _extract_roi(img, mid1, mid2, C, thumb, hand_type):
    vec   = np.array(mid2) - np.array(mid1)
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    if hand_type.lower() == "right":
        if np.dot(vec, np.array(thumb) - np.array(C)) > 0:
            angle += 180
    else:  # left / unknown
        if np.dot(vec, np.array(thumb) - np.array(C)) < 0:
            angle += 180
    side = np.linalg.norm(vec) * 2.5
    rect = (C, (side, side), angle)
    M = cv2.getRotationMatrix2D(C, angle, 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    roi = cv2.getRectSubPix(rot, (int(side), int(side)), C)
    box = cv2.boxPoints(rect).astype(int)
    return roi, box

# ----------------------------------------------------------------------
# Public function
# ----------------------------------------------------------------------

def extract_palm_roi(image_bgr):
    """
    Parameters
    ----------
    image_bgr : np.ndarray   BGR image.

    Returns
    -------
    roi_bgr, annotated_bgr, hand_type
        If landmarks fail → returns (None, None, None)
    """
    lms, hand_type = _run_mp_hands(image_bgr)
    if lms is None:
        return None, None, None

    # compute the 5 required points
    idx = lambda i: lms[i]
    mids4 = _midpoints([(idx(17), idx(18)),
                        (idx(14), idx(13)),
                        (idx(10), idx( 9)),
                        (idx( 6), idx( 5))])  # p17_18 … p6_5
    adj  = _midpoints([(mids4[0], mids4[1]),
                       (mids4[1], mids4[2]),
                       (mids4[2], mids4[3])])
    roi_mid1 = ((adj[0][0] + adj[1][0]) / 2, (adj[0][1] + adj[1][1]) / 2)
    roi_mid2 = ((adj[1][0] + adj[2][0]) / 2, (adj[1][1] + adj[2][1]) / 2)
    thumb    = idx(2)
    C        = _calculate_point_c(roi_mid1, roi_mid2, thumb)

    roi, box = _extract_roi(image_bgr, roi_mid1, roi_mid2, C, thumb, hand_type)

    # annotated image (optional)
    ann = image_bgr.copy()
    for i, (x, y) in enumerate(lms):
        cv2.circle(ann, (x, y), 3, (0, 255, 0), -1)
    cv2.polylines(ann, [box], True, (0, 255, 0), 2)
    cv2.circle(ann, tuple(map(int, C)), 6, (0, 0, 255), -1)

    return roi, ann, hand_type
