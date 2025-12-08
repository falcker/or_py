"""
Synthetic tank-oil generator WITH segmentation masks.

Input:
    - Folder of clean tank images (ideally oil-free)
    - oil_only_black_stain.png : extracted oil texture (RGBA, alpha = oil)

Output (per clean image):
    - <stem>_oil_XX.jpg       : RGB contaminated image
    - <stem>_oil_XX_mask.png  : binary mask (255 = oil, 0 = background)
"""

import random
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


# ---------------------------
# 0. Utility
# ---------------------------


def load_rgba(path):
    """Load image as RGBA."""
    return Image.open(path).convert("RGBA")


# ---------------------------
# 1. Oil sampler
# ---------------------------


class OilSampler:
    """
    Samples realistic oil patches from an oil-only RGBA texture,
    such as 'oil_only_black_stain.png'.
    """

    def __init__(self, oil_png_path):
        self.oil_img = load_rgba(oil_png_path)
        self.width, self.height = self.oil_img.size

        alpha = np.array(self.oil_img.split()[-1])
        # Coordinates of true oil pixels (row, col) = (y, x)
        ys, xs = np.where(alpha > 0)
        self.coords = np.column_stack([xs, ys])  # (x, y)
        self.alpha = alpha

        if len(self.coords) == 0:
            raise ValueError("No non-zero alpha pixels found in oil texture.")

    def sample_patch(
        self, min_size=80, max_size=260, min_fill_ratio=0.15, max_tries=30
    ):
        """
        Sample a rectangular patch that actually contains oil.

        min_size / max_size are in pixels for both width & height.
        min_fill_ratio = required fraction of non-zero alpha in the patch.
        """
        w, h = self.width, self.height

        for _ in range(max_tries):
            # Pick a random oil pixel as patch center
            x_center, y_center = self.coords[random.randrange(len(self.coords))]
            patch_w = random.randint(min_size, max_size)
            patch_h = random.randint(min_size, max_size)

            x1 = max(0, x_center - patch_w // 2)
            y1 = max(0, y_center - patch_h // 2)
            x2 = min(w, x1 + patch_w)
            y2 = min(h, y1 + patch_h)

            patch = self.oil_img.crop((x1, y1, x2, y2))
            patch_alpha = np.array(patch.split()[-1])
            fill_ratio = (patch_alpha > 0).mean()

            if fill_ratio >= min_fill_ratio:
                return patch

        # Fallback: a small dense patch around a random oil pixel
        x_center, y_center = self.coords[random.randrange(len(self.coords))]
        patch_w = min_size
        patch_h = min_size
        x1 = max(0, x_center - patch_w // 2)
        y1 = max(0, y_center - patch_h // 2)
        x2 = min(w, x1 + patch_w)
        y2 = min(h, y1 + patch_h)
        return self.oil_img.crop((x1, y1, x2, y2))


# ---------------------------
# 2. Patch augmentations
# ---------------------------


def jitter_patch_intensity(patch_rgba, value_range=(0.85, 1.15)):
    """
    Slightly brightens/darkens the oil while preserving color balance.
    value_range controls multiplicative brightness factor.
    """
    arr = np.array(patch_rgba).astype(np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3:]

    factor = random.uniform(*value_range)
    rgb = np.clip(rgb * factor, 0, 255)

    arr[..., :3] = rgb
    arr[..., 3:] = alpha
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def augment_patch_geometric(
    patch_rgba, scale_range=(0.6, 1.4), rotation_range=(-12, 12), allow_flip=True
):
    """
    Apply mild geometric jitter:
    - random uniform scaling
    - small rotation
    - optional random horizontal flip
    """
    w, h = patch_rgba.size

    # Scale
    scale = random.uniform(*scale_range)
    new_w = max(4, int(w * scale))
    new_h = max(4, int(h * scale))
    patch = patch_rgba.resize((new_w, new_h), resample=Image.BICUBIC)

    # Flip
    if allow_flip and random.random() < 0.5:
        patch = ImageOps.mirror(patch)

    # Rotate slightly, keeping size (no expand)
    angle = random.uniform(*rotation_range)
    patch = patch.rotate(angle, resample=Image.BICUBIC, expand=False)

    return patch


# ---------------------------
# 3. Placement utilities
# ---------------------------


def random_position(
    base_w,
    base_h,
    patch_w,
    patch_h,
    placement_mask=None,
    min_mask_fraction=0.8,
    max_tries=40,
):
    """
    Sample a random (x, y) position for the top-left of the patch.

    If placement_mask is provided (H x W bool),
    only place patches where most of the patch lies inside the mask.
    """
    if placement_mask is None:
        x = random.randint(0, max(0, base_w - patch_w))
        y = random.randint(0, max(0, base_h - patch_h))
        return x, y

    mask = placement_mask
    H, W = mask.shape

    for _ in range(max_tries):
        x = random.randint(0, max(0, base_w - patch_w))
        y = random.randint(0, max(0, base_h - patch_h))

        x2 = min(W, x + patch_w)
        y2 = min(H, y + patch_h)
        sub = mask[y:y2, x:x2]

        if sub.size == 0:
            continue
        if sub.mean() >= min_mask_fraction:
            return x, y

    # Fallback if mask is too restrictive
    x = random.randint(0, max(0, base_w - patch_w))
    y = random.randint(0, max(0, base_h - patch_h))
    return x, y


def build_simple_seal_band_mask(
    base_img, inner_radius_frac=0.78, outer_radius_frac=0.98, center=None
):
    """
    Very simple radial band mask around the roof perimeter for
    top-down tanks. Assumes image roughly centered on tank.

    Returns: mask (H x W, bool)
    """
    w, h = base_img.size
    cx, cy = center or (w / 2.0, h / 2.0)

    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = rr.max()

    inner = inner_radius_frac * r_max
    outer = outer_radius_frac * r_max

    mask = (rr >= inner) & (rr <= outer)
    return mask


# ---------------------------
# 4. Compositing + mask creation
# ---------------------------


def composite_oil_on_clean_with_mask(
    clean_img, oil_sampler, n_patches_range=(10, 25), seal_band_mask=None
):
    """
    Generate a new contaminated image and its binary oil mask.

    clean_img       : Pillow Image (RGB or RGBA)
    oil_sampler     : OilSampler instance
    n_patches_range : (min, max) number of patches to place
    seal_band_mask  : optional H x W bool mask restricting placement to seal band

    Returns:
        contaminated_rgba : RGBA image
        mask_img          : L image, 255 = oil, 0 = background
    """
    base = clean_img.convert("RGBA")
    base_w, base_h = base.size

    # mask array in [0, 255]
    mask_arr = np.zeros((base_h, base_w), dtype=np.uint8)

    n_patches = random.randint(*n_patches_range)

    for _ in range(n_patches):
        # 1) Sample raw patch from oil texture
        patch = oil_sampler.sample_patch(min_size=80, max_size=260, min_fill_ratio=0.15)

        # 2) Photometric jitter (brightness)
        patch = jitter_patch_intensity(patch, value_range=(0.9, 1.1))

        # 3) Geometric jitter (scale, rotate, flip)
        patch = augment_patch_geometric(
            patch, scale_range=(0.7, 1.4), rotation_range=(-15, 15), allow_flip=True
        )

        patch_alpha = np.array(patch.split()[-1])
        if patch_alpha.mean() < 5:  # skip almost-empty patches
            continue

        pw, ph = patch.size

        # 4) Choose placement (ideally near the seal)
        x, y = random_position(
            base_w, base_h, pw, ph, placement_mask=seal_band_mask, min_mask_fraction=0.7
        )

        # Clip in case we hit edges
        x = max(0, min(x, base_w - pw))
        y = max(0, min(y, base_h - ph))

        # 5) Composite onto base roof
        base.alpha_composite(patch, (x, y))

        # 6) Update mask (anything with alpha > 0 is oil)
        patch_mask = (patch_alpha > 0).astype(np.uint8) * 255
        y2 = y + ph
        x2 = x + pw
        sub = mask_arr[y:y2, x:x2]
        # OR operation so overlapping patches stay marked
        sub |= patch_mask
        mask_arr[y:y2, x:x2] = sub

    mask_img = Image.fromarray(mask_arr, mode="L")
    return base, mask_img


# ---------------------------
# 5. Batch driver
# ---------------------------


def generate_batch_with_masks(
    clean_dir, oil_png_path, out_img_dir, out_mask_dir, images_per_clean=10
):
    """
    clean_dir        : directory with clean tank images (JPG/PNG)
    oil_png_path     : oil-only texture (e.g. oil_only_black_stain.png)
    out_img_dir      : output directory for contaminated RGB images
    out_mask_dir     : output directory for binary masks (PNG)
    images_per_clean : how many augmented versions per clean image
    """
    out_img_dir = Path(out_img_dir)
    out_mask_dir = Path(out_mask_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    sampler = OilSampler(oil_png_path)
    clean_dir = Path(clean_dir)

    clean_paths = [
        p for p in clean_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    for clean_path in clean_paths:
        clean_img = Image.open(clean_path).convert("RGB")

        # Optional: build a rough seal band mask per image
        seal_mask = build_simple_seal_band_mask(
            clean_img, inner_radius_frac=0.78, outer_radius_frac=0.98
        )

        for i in range(images_per_clean):
            contaminated_rgba, mask_img = composite_oil_on_clean_with_mask(
                clean_img, sampler, n_patches_range=(12, 30), seal_band_mask=seal_mask
            )

            # Save RGB image
            img_name = f"{clean_path.stem}_oil_{i:02d}.jpg"
            img_path = out_img_dir / img_name
            contaminated_rgba.convert("RGB").save(img_path, quality=95)

            # Save mask (PNG to keep crisp edges)
            mask_name = f"{clean_path.stem}_oil_{i:02d}_mask.png"
            mask_path = out_mask_dir / mask_name
            mask_img.save(mask_path)

            print("saved", img_path, "and", mask_path)


# ---------------------------
# 6. Example usage
# ---------------------------

if __name__ == "__main__":
    clean_tanks_folder = r"C:\Users\Gebruiker\Documents\Falcker\AI\data\OlieDetectie\SyntheticDataset\original unedited images\Voor ChatGPT"
    oil_png_path = r"C:\Users\Gebruiker\Documents\Falcker\AI\data\Synthetiseren\Affinity\oil_extracted_base_clean.png"
    output_root_dir = Path(
        r"C:\Users\Gebruiker\Documents\Falcker\AI\data\OlieDetectie\SyntheticDataset\original unedited images\Voor ChatGPT"
    )
    print("Starting synthetic oil image generation...")
    generate_batch_with_masks(
        clean_dir=clean_tanks_folder,  # folder with clean roofs
        oil_png_path=oil_png_path,  # your extracted oil texture
        out_img_dir=output_root_dir / "synthetic_oil_images/",
        out_mask_dir=output_root_dir / "synthetic_oil_masks/",
        images_per_clean=20,
    )
    print("Synthetic oil image generation complete.")
