"""
augment_landmarks_corrected.py

For each folder in DATA_DIR:
 - load landmarks.json (or aug_annotations.json)
 - for each sample: apply N random augmentations (affine + photometric)
 - transform the 7 landmark points with the exact same affine matrix
 - save augmented image and add an entry into the same JSON file

This keeps images and landmarks aligned.
"""
import os
import json
import cv2
import numpy as np
import random
from glob import glob
from tqdm import tqdm

DATA_DIR = "augmented"           # root with gesture subfolders
OUT_JSON_NAMES = ("aug_annotations.json", "landmarks.json", "landmarks.json")
AUG_PER_IMAGE = 6                # how many augmented variants to create per original

# photometric augmentation helpers
def random_brightness(img, low=0.7, high=1.3):
    alpha = random.uniform(low, high)
    img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return img

def random_blur(img):
    k = random.choice([0, 0, 1, 1, 3])  # most of the time no blur
    if k <= 1:
        return img
    return cv2.GaussianBlur(img, (k|1, k|1), 0)

def random_noise(img, sigma=6):
    if sigma <= 0:
        return img
    noise = np.random.normal(0, random.uniform(0, sigma), img.shape).astype(np.int16)
    out = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out

# geometric transform builder (affine)
def random_affine(center, max_rot=20, scale_range=(0.9,1.1), max_trans=0.08, do_flip=True):
    cx, cy = center
    angle = random.uniform(-max_rot, max_rot)
    scale = random.uniform(scale_range[0], scale_range[1])
    # base rotation+scale matrix (2x3)
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)  # shape (2,3)
    # translation
    tx = random.uniform(-max_trans, max_trans) * cx * 2
    ty = random.uniform(-max_trans, max_trans) * cy * 2
    M[0,2] += tx
    M[1,2] += ty

    flipped = False
    if do_flip and random.random() < 0.5:
        # incorporate horizontal flip around the image center:
        # flip matrix: x' = -1*(x - cx) + cx  => multiply by [-1, 0; 0,1] and add 2*cx
        F = np.array([[-1, 0, 2*cx],
                      [ 0, 1, 0 ]], dtype=np.float32)
        M = F @ np.vstack([M, [0,0,1]])[:2,:]  # compose
        flipped = True

    return M, flipped

def apply_affine_to_points(pts, M):
    """
    pts: array shape (N,2) in pixel coords
    M: 2x3 affine matrix
    returns transformed pts (N,2)
    """
    pts_h = np.concatenate([pts, np.ones((pts.shape[0],1), dtype=np.float32)], axis=1)  # (N,3)
    out = (M @ pts_h.T).T
    return out

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    # normalize structure
    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError("Unsupported JSON structure: " + path)
    return samples, data  # return list and original container

def save_json(container, path):
    with open(path, "w") as f:
        json.dump(container, f, indent=2)

def augment_folder(folder, aug_per_image=AUG_PER_IMAGE):
    # find a JSON annotation file
    json_file = None
    for name in ("aug_annotations.json", "landmarks.json", "landmarks.json"):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            json_file = p
            break
    if json_file is None:
        # fallback: first .json found
        files = glob(os.path.join(folder, "*.json"))
        if files:
            json_file = files[0]
    if json_file is None:
        print(f"[WARN] no JSON annotation found in {folder}, skipping")
        return

    samples, container = load_json(json_file)
    if isinstance(container, dict) and "samples" not in container:
        container = {"samples": samples}
    elif isinstance(container, list):
        container = samples[:]  # treat as list, will rewrite

    new_entries = []
    for entry in tqdm(samples, desc=os.path.basename(folder)):
        fname = entry.get("file") or entry.get("image") or entry.get("img")
        pts = entry.get("landmarks") or entry.get("points") or entry.get("keypoints")
        if fname is None or pts is None:
            continue
        img_path = os.path.join(folder, fname)
        if not os.path.exists(img_path):
            # try raw fname
            if os.path.exists(fname):
                img_path = fname
            else:
                # skip missing image
                print(f"[WARN] missing image {img_path}, skipping")
                continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        # ensure points shape Nx2 (pixel coords or normalized)
        pts_arr = np.array(pts, dtype=np.float32)
        if pts_arr.ndim != 2 or pts_arr.shape[0] != 7 or pts_arr.shape[1] != 2:
            # invalid shape
            print(f"[WARN] invalid landmarks for {fname}, skipping")
            continue

        H, W = img.shape[:2]
        # if points look normalized (all <=1.01), convert to pixel coords for transform
        if np.all(pts_arr <= 1.01):
            pts_pix = np.zeros_like(pts_arr)
            pts_pix[:,0] = pts_arr[:,0] * W
            pts_pix[:,1] = pts_arr[:,1] * H
        else:
            pts_pix = pts_arr.copy()

        # Create augmented variants
        for ai in range(aug_per_image):
            # random affine transform centered on the image center
            M, flipped = random_affine(center=(W/2.0, H/2.0),
                                      max_rot=20,
                                      scale_range=(0.9, 1.1),
                                      max_trans=0.08,
                                      do_flip=True)
            # apply transform to image
            aug_img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # transform points
            pts_t = apply_affine_to_points(pts_pix, M)  # (7,2)

            # photometric
            aug_img = random_brightness(aug_img, 0.75, 1.25)
            aug_img = random_blur(aug_img)
            aug_img = random_noise(aug_img, sigma=6)

            # clip points to inside image
            pts_t[:,0] = np.clip(pts_t[:,0], 0, W-1)
            pts_t[:,1] = np.clip(pts_t[:,1], 0, H-1)

            # Save augmented file
            base, ext = os.path.splitext(fname)
            out_name = f"aug_{ai}_{base}{ext}"
            out_path = os.path.join(folder, out_name)
            cv2.imwrite(out_path, aug_img)

            # Save normalized landmarks (0..1)
            pts_norm = (pts_t.copy())
            pts_norm[:,0] = pts_norm[:,0] / float(W)
            pts_norm[:,1] = pts_norm[:,1] / float(H)
            pts_list = pts_norm.tolist()

            new_entries.append({"file": out_name, "landmarks": pts_list})

    # append new entries to container appropriately
    if isinstance(container, dict) and "samples" in container:
        container["samples"].extend(new_entries)
        save_json(container, json_file)
    elif isinstance(container, list):
        container.extend(new_entries)
        save_json(container, json_file)
    else:
        # fallback: write out aug_annotations.json
        outp = {"samples": new_entries}
        save_json(outp, os.path.join(folder, "aug_annotations.json"))

    print(f"[INFO] Augmented {len(new_entries)} samples in {folder}")

def main():
    for sub in sorted(os.listdir(DATA_DIR)):
        folder = os.path.join(DATA_DIR, sub)
        if not os.path.isdir(folder):
            continue
        augment_folder(folder, AUG_PER_IMAGE)

if __name__ == "__main__":
    main()
