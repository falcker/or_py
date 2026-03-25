import os
import shutil
from pathlib import Path

import torch
import clip
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm


# -----------------------------
# CONFIG
# -----------------------------
IMAGE_DIR = "input_images"
OUTPUT_DIR = "filtered_images"

# Similarity control
EPS = 0.1  # smaller = stricter clustering (try 0.05–0.2)
MIN_SAMPLES = 1  # keep as 1

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)


# -----------------------------
# LOAD IMAGES
# -----------------------------
def load_images(image_dir):
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(Path(image_dir).glob(ext))
    return sorted(image_paths)


# -----------------------------
# COMPUTE EMBEDDINGS
# -----------------------------
def compute_embeddings(image_paths):
    embeddings = []

    for path in tqdm(image_paths, desc="Encoding images"):
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = model.encode_image(image)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            embeddings.append(emb.cpu().numpy()[0])

        except Exception as e:
            print(f"Error processing {path}: {e}")
            embeddings.append(None)

    return np.array(embeddings, dtype=object)


# -----------------------------
# FILTER VALID EMBEDDINGS
# -----------------------------
def filter_valid(image_paths, embeddings):
    valid_paths = []
    valid_embeddings = []

    for p, e in zip(image_paths, embeddings):
        if e is not None:
            valid_paths.append(p)
            valid_embeddings.append(e)

    return valid_paths, np.vstack(valid_embeddings)


# -----------------------------
# CLUSTER
# -----------------------------
def cluster_embeddings(embeddings):
    print("Clustering images...")
    clustering = DBSCAN(
        eps=EPS,
        min_samples=MIN_SAMPLES,
        metric="cosine"
    ).fit(embeddings)

    return clustering.labels_


# -----------------------------
# SELECT REPRESENTATIVES
# -----------------------------
def select_representatives(image_paths, embeddings, labels):
    clusters = {}

    for path, emb, label in zip(image_paths, embeddings, labels):
        clusters.setdefault(label, []).append((path, emb))

    selected = []

    for label, items in clusters.items():
        # Compute centroid
        embs = np.array([x[1] for x in items])
        centroid = embs.mean(axis=0)

        # Pick closest to centroid
        best_item = min(items, key=lambda x: np.linalg.norm(x[1] - centroid))
        selected.append(best_item[0])

    return selected


# -----------------------------
# COPY RESULTS
# -----------------------------
def copy_selected(selected_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for path in selected_paths:
        dst = Path(output_dir) / path.name
        shutil.copy(path, dst)


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading images...")
    image_paths = load_images(IMAGE_DIR)

    print(f"Found {len(image_paths)} images")

    embeddings = compute_embeddings(image_paths)

    image_paths, embeddings = filter_valid(image_paths, embeddings)

    print(f"{len(image_paths)} valid images")

    labels = cluster_embeddings(embeddings)

    selected = select_representatives(image_paths, embeddings, labels)

    print(f"Selected {len(selected)} representative images")

    copy_selected(selected, OUTPUT_DIR)

    print(f"Done! Filtered images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()