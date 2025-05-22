# -*- coding: utf-8 -*-
"""
Jetson listener for NFS-triggered palm-print pipeline
Python 3.6-compatible
"""
import os, re, json, time, pathlib, queue, threading, logging, collections
from typing import Tuple, Dict, List

import numpy as np
import cv2
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import torch
import torchvision.transforms as T

# ────────── paths & constants ──────────────────────────────────────────
IN_DIR   = pathlib.Path("/mnt/jetson_cam/in")
OUT_DIR  = pathlib.Path("/mnt/jetson_cam/out")
DATA_DIR = pathlib.Path("data")
ROI_DIR  = pathlib.Path("/mnt/jetson_cam/roi")
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

logging.info("Loading DINOv2 ViT-S/14 …")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                       pretrained=True)
model.eval().to(device)
if device.type == "cuda":
    model.half()

transform = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_features(img_path: str) -> np.ndarray:
    """Return L2-normalised 384-D feature vector for one image."""
    pil = Image.open(img_path).convert("L")        # grayscale
    pil = Image.merge("RGB", (pil, pil, pil))      # fake 3-ch
    ten = transform(pil).unsqueeze(0).to(device)
    if device.type == "cuda":
        ten = ten.half()
    vec = model(ten).cpu().numpy().flatten()
    vec /= np.linalg.norm(vec)
    return vec.astype(np.float32)

# ────────── gallery I/O ────────────────────────────────────────────────
def load_gallery() -> Tuple[np.ndarray, Dict[str, int]]:
    if NEW_MATRIX.exists():
        A = np.load(NEW_MATRIX)
    elif FEATURE_NPY.exists():
        A = np.load(FEATURE_NPY)
    else:
        A = np.empty((DIM, 0), np.float32)
    classes = json.load(CLASS_JSON.open()) if CLASS_JSON.exists() else {}
    return A, classes

def save_gallery(A: np.ndarray, classes: Dict[str, int]) -> None:
    np.save(NEW_MATRIX, A)
    with CLASS_JSON.open("w") as f:
        json.dump(classes, f, indent=2)

A, class_dict = load_gallery()
logging.info("Gallery columns (subjects): %d", A.shape[1])

# ────────── import original helpers ────────────────────────────────────
from enroll   import run_enrollment   # expected to return (A, class_dict)
from identify import run_identification

# ────────── helper: warm-up DINOv2 once ────────────────────────────────
def warm_dinov2() -> None:
    with torch.no_grad():
        dummy = torch.zeros((1, 3, 224, 224), device=device)
        if device.type == "cuda":
            dummy = dummy.half()
        _ = model(dummy)

warm_dinov2()

# ────────── session state ──────────────────────────────────────────────
enr_count: collections.Counter = collections.Counter()  # {sid: processed}
enr_label: Dict[str, str]       = {}                    # {sid: label}
id_votes : Dict[str, List[str]] = collections.defaultdict(list)

# ────────── helper to write JSON results ───────────────────────────────
def write_json(stem: str, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp   = OUT_DIR / (stem + ".tmp")
    final = OUT_DIR / (stem + ".ok.json")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.rename(final)

# ────────── directory monitor (watchdog) ───────────────────────────────
work_q: queue.Queue = queue.Queue(maxsize=256)

class NewFile(FileSystemEventHandler):
    """Enqueue every .jpg that appears via atomic rename in IN_DIR."""
    def on_moved(self, event):
        # event.dest_path is the *new* name after rename
        if not event.is_directory and event.dest_path.endswith(".jpg"):
            work_q.put(pathlib.Path(event.dest_path))
    def on_created(self, event):
        # created events will fire if the producer writes directly,
        # but we still only accept ready-made .jpg
        if (not event.is_directory
                and event.src_path.endswith(".jpg")):
            work_q.put(pathlib.Path(event.src_path))

# ────────── ROI extraction helper ──────────────────────────────────────
from google_roi import extract_palm_roi    # your landmark/ROI function

def wait_for_stable(path: pathlib.Path, retries: int = 20,
                    pause: float = 0.05) -> bool:
    """Return True when file size has stopped changing."""
    prev = -1
    for _ in range(retries):
        try:
            size = path.stat().st_size
            if size == prev and size > 0:
                return True
            prev = size
        except FileNotFoundError:
            pass
        time.sleep(pause)
    return False

def roi_from_image(img_path: pathlib.Path):
    # ensure file is fully written (if direct write, not rename)
    wait_for_stable(img_path, retries=10, pause=0.1)

    img = cv2.imread(str(img_path))
    if img is None:
        logging.error("OpenCV failed to read %s", img_path.name)
        return None

    roi, _, _ = extract_palm_roi(img)
    if roi is None:
        logging.warning("Landmarks not found in %s", img_path.name)
        return None

    roi_path = ROI_DIR / (img_path.stem + "_roi.jpg")
    cv2.imwrite(str(roi_path), roi)
    return roi_path

# ────────── worker thread ──────────────────────────────────────────────
def process_loop():
    global A, class_dict
    while True:
        p: pathlib.Path = work_q.get()
        # ignore any stray .tmp or unrelated file
        if p.suffix != ".jpg":
            continue

        m = FNAME_RE.match(p.name)
        if not m:
            logging.warning("Ignoring bad filename %s", p.name)
            try: p.unlink()
            except FileNotFoundError: pass
            continue

        mode, sid, idx = m.group(1).upper(), m.group(2), m.group(3)

        try:
            roi_path = roi_from_image(p)
            if roi_path is None:
                continue

            if mode == "ENR":
                A, class_dict = run_enrollment(
                    A, class_dict, NEW_MATRIX, extract_features, str(roi_path)
                )
                if sid not in enr_label:
                    # assign label for this new subject
                    enr_label[sid] = max(class_dict, key=class_dict.get)
                enr_count[sid] += 1

                if enr_count[sid] == BATCH:
                    save_gallery(A, class_dict)
                    write_json(f"ENR_{sid}", {
                        "mode":    "enroll",
                        "session": sid,
                        "label":   enr_label[sid],
                        "images":  BATCH
                    })
                    enr_count.pop(sid, None)
                    enr_label.pop(sid, None)

            else:  # IDENTIFY
                label = run_identification(
                    A, class_dict, extract_features, str(roi_path)
                )
                if label:
                    id_votes[sid].append(label)

                if len(id_votes[sid]) == BATCH:
                    votes   = collections.Counter(id_votes.pop(sid))
                    top_cnt = votes.most_common(1)[0][1]
                    winners = [lbl for lbl, c in votes.items() if c == top_cnt]
                    final   = winners[0]  # tie-break: first winner
                    write_json(f"ID_{sid}", {
                        "mode":    "identify",
                        "session": sid,
                        "votes":   dict(votes),
                        "winner":  final,
                        "count":   top_cnt
                    })
                    logging.info("[✓] Final result for session %s → %s",
                                 sid, final)

        finally:
            # clean up original file if still present
            try: p.unlink()
            except FileNotFoundError:
                pass

# ────────── main ───────────────────────────────────────────────────────
def main():
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    observer = Observer()
    observer.schedule(NewFile(), str(IN_DIR), recursive=False)
    observer.start()

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
