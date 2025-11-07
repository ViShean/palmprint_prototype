# -*- coding: utf-8 -*-
"""
Jetson listener for NFS‑triggered palm‑print pipeline (ROI‑ready images)
Python 3.6‑compatible

▶ **Atomic batch enrollment** (6 ROI images → one participant)
▶ **Incrementing participant IDs (001, 002, …) automatically)**
▶ **CLASS_JSON maps each participant to all its column indices**

**NOTE:** Logging calls have been replaced with simple `print()` statements
per user request.
"""

import os, re, json, time, pathlib, queue, threading, collections
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import torch
import torchvision.transforms as T

# ────────── paths & constants ──────────────────────────────────────────
IN_DIR   = pathlib.Path("/home/nemo/server/in")
OUT_DIR  = pathlib.Path("/home/nemo/server/out")
DATA_DIR = pathlib.Path("data")

FEATURE_NPY = DATA_DIR / "feature_matrix_s14_no_interpolate.npy"
CLASS_JSON  = DATA_DIR / "feature_matrix_s14_no_interpolate.json"
NEW_MATRIX  = DATA_DIR / "test.npy"

DIM   = 384
BATCH = 6  # images per participant
FNAME_RE = re.compile(r"^(ENR|ID)_(\d+)_0[0-5]\.jpg$", re.I)  # 00‑05 only

# ────────── utility helpers ───────────────────────────────────────────

def _safe_unlink(path: pathlib.Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass

def _next_pid(cls_dict: Dict[str, List[int]]) -> str:
    if not cls_dict:
        return "001"
    nums = [int(k) for k in cls_dict.keys() if k.isdigit()]
    return f"{max(nums)+1:03d}"

# ────────── DINOv2 model ──────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()

print("Loading DINOv2 ViT‑S/14 …")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
model.eval().to(device)
if device.type == "cuda":
    model.half()

transform = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_features(img_path: str) -> np.ndarray:
    pil = Image.open(img_path).convert("L")
    pil = Image.merge("RGB", (pil, pil, pil))
    ten = transform(pil).unsqueeze(0).to(device)
    if device.type == "cuda":
        ten = ten.half()
    vec = model(ten).cpu().numpy().flatten()
    vec /= np.linalg.norm(vec)
    return vec.astype(np.float32)

# ────────── gallery I/O ────────────────────────────────────────────────

def load_gallery() -> Tuple[np.ndarray, Dict[str, List[int]]]:
    if NEW_MATRIX.exists():
        A = np.load(NEW_MATRIX)
    elif FEATURE_NPY.exists():
        A = np.load(FEATURE_NPY)
    else:
        A = np.empty((DIM, 0), np.float32)
    if CLASS_JSON.exists():
        with CLASS_JSON.open() as f:
            cls = json.load(f)
    else:
        cls = {}
    for k, v in list(cls.items()):
        if isinstance(v, int):
            cls[k] = [v]
    return A, cls

def save_gallery(A: np.ndarray, cls: Dict[str, List[int]]) -> None:
    np.save(NEW_MATRIX, A)
    with CLASS_JSON.open("w") as f:
        json.dump(cls, f, indent=2)

A, class_dict = load_gallery()
print(f"Gallery columns: {A.shape[1]} • Participants: {len(class_dict)}")

# ────────── import original helpers ────────────────────────────────────
from enroll   import run_enrollment
from identify import run_identification

# ────────── warm‑up DINOv2 ────────────────────────────────────────────

def _warm():
    with torch.no_grad():
        dummy = torch.zeros((1, 3, 224, 224), device=device)
        if device.type == "cuda":
            dummy = dummy.half()
        _ = model(dummy)

_warm()

# ────────── session state ──────────────────────────────────────────────

enr_paths: Dict[str, List[pathlib.Path]] = collections.defaultdict(list)
enr_count: collections.Counter = collections.Counter()
id_votes : Dict[str, List[str]] = collections.defaultdict(list)

# ────────── NEW HELPERS (put near other helpers) ──────────────────────
def _abandon_incomplete_enroll(curr_sid: str) -> None:
    """Drop any ENR sessions ≠ curr_sid that never reached BATCH images."""
    for sid, paths in list(enr_paths.items()):
        if sid == curr_sid or len(paths) >= BATCH:
            continue
        print(f"[ABORT] ENR {sid}: only {len(paths)}/{BATCH} images — discarding")
        for pt in paths:
            _safe_unlink(pt)
        enr_paths.pop(sid, None)
        enr_count.pop(sid, None)

def _abandon_incomplete_ident(curr_sid: str) -> None:
    """Drop any IDENTIFY sessions ≠ curr_sid that never reached BATCH votes."""
    for sid, votes in list(id_votes.items()):
        if sid == curr_sid or len(votes) >= BATCH:
            continue
        print(f"[ABORT] ID  {sid}: only {len(votes)}/{BATCH} votes — discarding")
        id_votes.pop(sid, None)
# ────────── write JSON helper ─────────────────────────────────────────

def write_json(stem: str, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp   = OUT_DIR / f"{stem}.tmp"
    final = OUT_DIR / f"{stem}.json"
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.rename(final)

# ────────── watchdog setup ────────────────────────────────────────────

work_q: queue.Queue = queue.Queue(maxsize=256)

class NewFile(FileSystemEventHandler):
    def on_moved(self, event):
        if not event.is_directory and event.dest_path.endswith(".jpg"):
            work_q.put(pathlib.Path(event.dest_path))
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".jpg"):
            work_q.put(pathlib.Path(event.src_path))

# ────────── wait until file stable ────────────────────────────────────

def wait_for_stable(path: pathlib.Path, retries: int = 20, pause: float = 0.05):
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

# ────────── representative dict for ID ────────────────────────────────

def _rep_dict(cls: Dict[str, List[int]]) -> Dict[str, List[int]]:
    return cls

# ────────── worker thread ──────────────────────────────────────────────

def process_loop():
    global A, class_dict
    while True:
        p: pathlib.Path = work_q.get()
        if p.suffix.lower() != ".jpg":
            continue

        m = FNAME_RE.match(p.name)
        if not m:
            print(f"[WARN] Ignoring bad filename {p.name}")
            _safe_unlink(p)
            continue

        mode, sid = m.group(1).upper(), m.group(2)
        wait_for_stable(p)

        if mode == "ENR":
            _abandon_incomplete_enroll(sid)
            enr_paths[sid].append(p)
            enr_count[sid] = len(enr_paths[sid])
            print(f"Image receied: {p}")
            print(f"→ ENR {sid}: {enr_count[sid]}/{BATCH} images buffered")

            if enr_count[sid] == BATCH:
                feats = [extract_features(str(pt)) for pt in enr_paths[sid]]
                start = A.shape[1]
                A = np.hstack([A, np.column_stack(feats)])
                cols = list(range(start, start+BATCH))

                pid = _next_pid(class_dict)
                class_dict.setdefault(pid, []).extend(cols)

                save_gallery(A, class_dict)
                write_json(f"results", {
                    "mode": "enroll",
                    "result": pid,
                })
                for pt in enr_paths.pop(sid):
                    _safe_unlink(pt)
                enr_count.pop(sid, None)

        else:  # IDENTIFY
            _abandon_incomplete_ident(sid)
            if A.shape[1] == 0:
                print(f"[WARN] No participants enrolled yet — skipping identify for {p.name}")
                _safe_unlink(p)
                continue
            rep_dict = _rep_dict(class_dict)              
            
            label = run_identification(A, rep_dict, extract_features, str(p))
            if label:
                id_votes[sid].append(label)

            # Once we have 6 votes, decide the winner
            if len(id_votes[sid]) == BATCH:
                votes   = collections.Counter(id_votes.pop(sid))
                top_cnt = votes.most_common(1)[0][1]
                winners = [lbl for lbl, c in votes.items() if c == top_cnt]
                final   = winners[0]
                print(f"[✓] IDENTIFY {sid}: winner → {final}  votes → {dict(votes)}")

                write_json(f"results", {
                    "mode":    "identify",
                    "session": sid,
                    "votes":   dict(votes),
                    "result":  final,
                    "count":   top_cnt
                })
            _safe_unlink(p)

# ────────── main ───────────────────────────────────────────────────────

def main():
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    observer = Observer()
    observer.schedule(NewFile(), str(IN_DIR), recursive=False)
    observer.start()

    threading.Thread(target=process_loop, daemon=True).start()
    print(f"Watching {IN_DIR} (batch size {BATCH})…  Ctrl-C to stop")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
