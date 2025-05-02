# -*- coding: utf-8 -*-
"""
Jetson NFS-triggered palm-print pipeline (Python 3.6).
Watches /srv/cam/in  ─► runs Darknet ROI crop ─► DINOv2 features
Enrols or identifies in batches of 7 images.
"""

import os, re, json, time, pathlib, queue, threading, logging, collections
import numpy as np
import cv2
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import torch, torchvision.transforms as T

# ────────── paths & constants ────────────────────────────────────────
IN_DIR   = pathlib.Path("/srv/cam/in")
OUT_DIR  = pathlib.Path("/srv/cam/out")
ROI_DIR  = pathlib.Path("/srv/cam/roi")
DATA_DIR = pathlib.Path("data")
ROI_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NPY = DATA_DIR / "feature_matrix_s14_no_interpolate.npy"
CLASS_JSON  = DATA_DIR / "feature_matrix_s14_no_interpolate.json"
NEW_MATRIX  = DATA_DIR / "test.npy"

DIM        = 384
BATCH      = 7
FNAME_RE   = re.compile(r"^(ENR|ID)_(\d+)_([0-6])\.jpg$", re.I)

# ────────── DINOv2 model ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

logging.info("Loading DINOv2 model …")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
model.eval().to(device)
if device.type == "cuda":
    model.half()

transform = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_features(img_path):
    pil = Image.open(img_path).convert("L")
    pil = Image.merge("RGB", (pil, pil, pil))
    ten = transform(pil).unsqueeze(0).to(device)
    if device.type == "cuda":
        ten = ten.half()
    vec = model(ten).cpu().numpy().flatten()
    vec /= np.linalg.norm(vec)
    return vec.astype(np.float32)

# ────────── gallery I/O ──────────────────────────────────────────────
def load_gallery():
    if NEW_MATRIX.exists():
        A = np.load(NEW_MATRIX)
    elif FEATURE_NPY.exists():
        A = np.load(FEATURE_NPY)
    else:
        A = np.empty((DIM, 0), np.float32)
    classes = json.load(CLASS_JSON.open()) if CLASS_JSON.exists() else {}
    return A, classes

def save_gallery(A, classes):
    np.save(NEW_MATRIX, A)
    with CLASS_JSON.open("w") as f:
        json.dump(classes, f, indent=2)

A, class_dict = load_gallery()
logging.info("Gallery columns: %d", A.shape[1])

# ────────── Darknet ROI helpers (unchanged) ──────────────────────────
from RoiExtraction import (run_darknet_detection, extract_points_from_detections,
                           find_closest_trio, calculate_midpoints,
                           calculate_point_c, extract_roi)

DARKNET_DIR   = pathlib.Path("/home/nemo/Documents/palmprint-authenticator/ROI_Extraction_Jetson/darknet")
OBJ_DATA      = pathlib.Path("/home/nemo/Documents/palmprint-authenticator/ROI_Extraction_Jetson/obj.data")
CFG_FILE      = pathlib.Path("/home/nemo/Documents/palmprint-authenticator/ROI_Extraction_Jetson/yolov3-tiny.cfg")
WEIGHTS_FILE  = pathlib.Path("/home/nemo/Documents/palmprint-authenticator/ROI_Extraction_Jetson/yolov3-tiny_final.weights")
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def roi_from_image(img_path):
    if img_path.suffix.lower() not in SUPPORTED_EXT:
        logging.warning("Bad extension %s", img_path.name)
        return None
    img = cv2.imread(str(img_path))
    if img is None:
        logging.error("OpenCV failed on %s", img_path)
        return None
    dets = run_darknet_detection(str(DARKNET_DIR), str(OBJ_DATA),
                                 str(CFG_FILE), str(WEIGHTS_FILE),
                                 str(img_path), str(img_path.parent))
    pts = extract_points_from_detections(dets)
    if len(pts) < 4:
        return None
    trio, gap = find_closest_trio(pts)
    if len(trio) < 3 or gap is None:
        return None
    mids   = calculate_midpoints(trio, gap)
    pc     = calculate_point_c(mids[0], mids[1], gap)
    roi, _ = extract_roi(img, mids, pc, gap, hand_type="right")
    roi_path = ROI_DIR / (img_path.stem + "_roi.jpg")
    cv2.imwrite(str(roi_path), roi)
    return roi_path

# ────────── enrol / identify helpers ─────────────────────────────────
from enroll   import run_enrollment
from identify import run_identification   # your existing identify.py

enr_count = collections.Counter()
enr_label = {}                # sid -> participant_id
id_votes  = collections.defaultdict(list)

def write_json(stem, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp   = OUT_DIR / (stem + ".tmp")
    final = OUT_DIR / (stem + ".ok.json")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.rename(final)

# ────────── watchdog queue ───────────────────────────────────────────
work_q = queue.Queue(maxsize=256)

class NewFile(FileSystemEventHandler):
    def on_moved(self, ev):
        if not ev.is_directory and ev.dest_path.endswith(".jpg"):
            work_q.put(pathlib.Path(ev.dest_path))
    def on_created(self, ev):
        if not ev.is_directory and ev.src_path.endswith(".jpg"):
            work_q.put(pathlib.Path(ev.src_path))

# ────────── worker thread ────────────────────────────────────────────
def process_loop():
    global A, class_dict
    while True:
        p = work_q.get()
        m = FNAME_RE.match(p.name)
        if not m:
            logging.warning("Ignoring %s", p.name)
            try: p.unlink()
            except: pass
            continue
        mode, sid, _ = m.group(1).upper(), m.group(2), m.group(3)
        try:
            roi_path = roi_from_image(p)
            if roi_path is None:
                continue
            if mode == "ENR":
                A, class_dict, pid = run_enrollment(
                    A, class_dict, NEW_MATRIX,
                    extract_features, str(roi_path),
                    enr_label.get(sid)
                )
                if pid is None:
                    continue
                enr_label.setdefault(sid, pid)
                enr_count[sid] += 1
                if enr_count[sid] == BATCH:
                    save_gallery(A, class_dict)
                    write_json("ENR_" + sid, {
                        "mode": "enroll",
                        "session": sid,
                        "label": pid,
                        "images": BATCH
                    })
                    enr_count.pop(sid, None)
                    enr_label.pop(sid, None)
            else:  # IDENTIFY
                label = run_identification(A, class_dict,
                                           extract_features, str(roi_path))
                if label:
                    id_votes[sid].append(label)
                if len(id_votes[sid]) == BATCH:
                    votes   = collections.Counter(id_votes.pop(sid))
                    max_cnt = votes.most_common(1)[0][1]
                    winners = [l for l, c in votes.items() if c == max_cnt]
                    final   = winners[0]
                    write_json("ID_" + sid, {
                        "mode": "identify",
                        "session": sid,
                        "votes": dict(votes),
                        "winner": final,
                        "count": max_cnt
                    })
                    logging.info("Session %s identified as %s", sid, final)
        finally:
            try: p.unlink()
            except: pass

# ────────── main ─────────────────────────────────────────────────────
def main():
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # warm-up
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        if device.type == "cuda":
            dummy = dummy.half()
        _ = model(dummy)

    obs = Observer()
    obs.schedule(NewFile(), str(IN_DIR), recursive=False)
    obs.start()
    threading.Thread(target=process_loop, daemon=True).start()

    logging.info("Watching %s (batch %d)… Ctrl-C to stop", IN_DIR, BATCH)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()

if __name__ == "__main__":
    main()
