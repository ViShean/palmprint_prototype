# -*- coding: utf-8 -*-
"""
Jetson listener for NFS-triggered palm-print pipeline
Python 3.6-compatible version
"""

import os, re, json, time, pathlib, queue, threading, logging, collections
import numpy as np
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch, torchvision.transforms as T

# ────────── paths & constants ──────────────────────────────────────────
IN_DIR   = pathlib.Path("/srv/cam/in")
OUT_DIR  = pathlib.Path("/srv/cam/out")
DATA_DIR = pathlib.Path("data")
ROI_DIR = pathlib.Path("/srv/cam/roi")   # <— choose any folder you like
ROI_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NPY = DATA_DIR / "feature_matrix_s14_no_interpolate.npy"
CLASS_JSON  = DATA_DIR / "feature_matrix_s14_no_interpolate.json"
NEW_MATRIX  = DATA_DIR / "test.npy"

DIM        = 384
BATCH      = 7
FNAME_RE   = re.compile(r"^(ENR|ID)_(\d+)_([0-6])\.jpg$", re.I)

# ────────── DINOv2 model ──────────────────────────────────────────────
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
def extract_features(img_path: str) -> np.ndarray:
    """Return L2-normalised 384-D feature vector for one image path."""
    pil = Image.open(img_path).convert("L")
    pil = Image.merge("RGB", (pil, pil, pil))
    ten = transform(pil).unsqueeze(0).to(device)
    if device.type == "cuda":
        ten = ten.half()
    vec = model(ten).cpu().numpy().flatten()
    vec /= np.linalg.norm(vec)
    return vec.astype(np.float32)

# ────────── gallery I/O ───────────────────────────────────────────────
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
logging.info("Gallery columns (subjects): %d", A.shape[1])

# ────────── import original helpers ───────────────────────────────────
from enroll   import run_enrollment
from identify import run_identification
# expected signatures:
#   A, class_dict = run_enrollment(A, class_dict, NEW_MATRIX, extract_features, img_path)
#   label         = run_identification(A, class_dict, extract_features, img_path)

# ────────── session buffers ──────────────────────────────────────────
enr_count = collections.Counter()           # {sid: processed_count}
enr_label = {}                              # {sid: label_assigned}
id_votes  = collections.defaultdict(list)   # {sid: [label, …]}

# ────────── helper to write JSON results ─────────────────────────────
def write_json(stem, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp   = OUT_DIR / (stem + ".tmp")
    final = OUT_DIR / (stem + ".ok.json")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.rename(final)

# ────────── watchdog setup ───────────────────────────────────────────
work_q = queue.Queue(maxsize=256)           # 3.6-friendly (no annotation)

class NewFile(FileSystemEventHandler):
    """Enqueue every .jpg that gets atomically moved into IN_DIR."""
    def on_moved(self, event):
        if not event.is_directory and event.dest_path.endswith(".jpg"):
            work_q.put(pathlib.Path(event.dest_path))
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".jpg"):
            work_q.put(pathlib.Path(event.src_path))
# ----------  ROI extraction with Darknet  (Py-3.6 friendly)  ----------
import cv2
import typing
from google_roi import extract_palm_roi

def roi_from_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        logging.error("OpenCV failed to read %s", img_path)
        return None
    roi, _, _ = extract_palm_roi(img)
    if roi is None:
        logging.warning("Landmarks not found in %s", img_path.name)
        return None
    roi_path = ROI_DIR / (img_path.stem + "_roi.jpg")
    cv2.imwrite(str(roi_path), roi)
    return roi_path


# ────────── worker thread ────────────────────────────────────────────
def process_loop():
    global A, class_dict
    while True:
        p = work_q.get()
        m = FNAME_RE.match(p.name)
        if not m:
            logging.warning("Ignoring bad filename %s", p.name)
            try: p.unlink()
            except FileNotFoundError: pass
            continue

        mode, sid, idx = m.group(1).upper(), m.group(2), m.group(3)

        try:
            if mode == "ENR":
                roi_path = roi_from_image(p)
                if roi_path is None:
                    continue         # skip this frame, do not count toward batch
                A, class_dict = run_enrollment(
                    A, class_dict, NEW_MATRIX, extract_features, str(roi_path)
                )
                # remember the label returned on first image
                if sid not in enr_label:
                    enr_label[sid] = max(class_dict, key=class_dict.get)
                enr_count[sid] += 1

                if enr_count[sid] == BATCH:
                    save_gallery(A, class_dict)
                    write_json("ENR_" + sid, {
                        "mode": "enroll",
                        "session": sid,
                        "label": enr_label[sid],
                        "images": BATCH
                    })
                    enr_count.pop(sid, None)
                    enr_label.pop(sid, None)

            else:  # IDENTIFY
                roi_path = roi_from_image(p)
                if roi_path is None:
                    continue
                label = run_identification(
                    A, class_dict, extract_features, str(roi_path)
                )
                if label:
                    id_votes[sid].append(label)
                if len(id_votes[sid]) == BATCH:
                    votes   = collections.Counter(id_votes.pop(sid))
                    max_cnt = votes.most_common(1)[0][1]
                    winners = [lbl for lbl, c in votes.items() if c == max_cnt]
                    final   = winners[0]   # simple tie-break rule
                    write_json("ID_" + sid, {
                        "mode": "identify",
                        "session": sid,
                        "votes": dict(votes),
                        "winner": final,
                        "count": max_cnt
                    })
                    print(f"[✓] Final result for session {sid} written as {final}")

        finally:
            try: p.unlink()
            except FileNotFoundError:
                pass

# ────────── main entry ───────────────────────────────────────────────
def main():
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    observer = Observer()
    observer.schedule(NewFile(), str(IN_DIR), recursive=False)
    observer.start()
    # Dummy call to Warm up the transform + model before first real use as the first image is always slower
    with torch.no_grad():
        dummy_image = Image.new("L", (256, 256))  # dummy grayscale image
        dummy_rgb = Image.merge("RGB", (dummy_image, dummy_image, dummy_image))
        dummy_tensor = transform(dummy_rgb).unsqueeze(0).to(device)
        if device.type == "cuda":
            dummy_tensor = dummy_tensor.half()
        _ = model(dummy_tensor)

    threading.Thread(target=process_loop, daemon=True).start()
    logging.info("Watching %s  (batch size %d)… Ctrl-C to stop",
                 IN_DIR, BATCH)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
