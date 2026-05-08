#!/usr/bin/env python3
"""
Visual Diff Detector — powered by Claude
=========================================
Detects visual differences between two or more images using the Anthropic API.

Usage (CLI):
    python claude_change_detect.py image1.jpg image2.jpg
    python claude_change_detect.py image1.jpg image2.jpg image3.jpg
    python claude_change_detect.py image1.jpg image2.jpg --api-key sk-ant-...
    python claude_change_detect.py image1.jpg image2.jpg --output result.jpg
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

from change_detection.prompts import COMPARISON_SYSTEM_PROMPT_LEAK_FOCUS

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    print("Warning: ANTHROPIC_API_KEY not set. CLI mode will not work without it.")


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def encode_image(path: str) -> tuple[str, str]:
    """Return (base64_data, mime_type) for an image file."""
    mime, _ = mimetypes.guess_type(path)
    if mime not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        mime = "image/jpeg"
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode(), mime


def build_image_block(path: str) -> dict:
    data, mime = encode_image(path)
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}


# ─────────────────────────────────────────────
#  Claude API
# ─────────────────────────────────────────────

def call_claude(image_paths: list[str], api_key: str, prompt: str) -> dict:
    """Send two or more images to Claude and return the parsed JSON result."""
    if len(image_paths) < 2:
        raise ValueError("At least two images are required.")

    content = [build_image_block(p) for p in image_paths] + [{"type": "text", "text": prompt}]

    payload = {
        "model": "claude-opus-4-7",
        "max_tokens": 100000,
        "messages": [{"role": "user", "content": content}],
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API error {e.code}: {e.read().decode()}") from e

    text = "".join(b.get("text", "") for b in data.get("content", [])).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse Claude response as JSON:\n{text}") from e


# ─────────────────────────────────────────────
#  Image annotation (Pillow)
# ─────────────────────────────────────────────

def annotate_image(img_path: str, result: dict, output_path: str) -> str:
    """Draw bounding box on the last image and save to output_path."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("⚠  Pillow not installed — skipping annotation. Install with: pip install pillow")
        return ""

    bb = result.get("bounding_box", {})
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    x, y, w, h = bb["x"], bb["y"], bb["width"], bb["height"]
    if x==0 and y==0 and w==0 and h==0:
        print("⚠  No bounding box detected — skipping annotation.")
        return ""
    lw = max(4, img.width // 400)

    draw.rectangle([x, y, x + w, y + h], fill=(255, 59, 48, 40))
    draw.rectangle([x, y, x + w, y + h], outline=(255, 59, 48, 255), width=lw)

    label = "anomaly detected"
    font_size = max(14, img.width // 120)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox_text = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
    pad = 6
    lx, ly = x, max(0, y - th - pad * 2)
    draw.rectangle([lx, ly, lx + tw + pad * 2, ly + th + pad * 2], fill=(255, 59, 48, 230))
    draw.text((lx + pad, ly + pad), label, fill=(255, 255, 255, 255), font=font)

    img.save(output_path, quality=92)
    return output_path


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def run_cli(args):
    key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("Error: provide --api-key or set ANTHROPIC_API_KEY.")
        sys.exit(1)

    missing = [p for p in args.images if not Path(p).exists()]
    if missing:
        print(f"Error: file(s) not found: {', '.join(missing)}")
        sys.exit(1)

    print(f"🔍  Sending {len(args.images)} image(s) to Claude...")
    result = call_claude(args.images, key, COMPARISON_SYSTEM_PROMPT_LEAK_FOCUS)

    print("\n── Result ──────────────────────────────")
    print(f"📋  Description : {result.get('description', 'N/A')}")
    bb = result.get("bounding_box", {})
    print(f"📦  Bounding box: x={bb.get('x')} y={bb.get('y')} w={bb.get('width')} h={bb.get('height')}")

    json_path = str(Path(args.output).with_suffix(".json"))
    with open(json_path, "w") as f:
        json.dump({"bounding_box": bb}, f, indent=2)
    print(f"\n✅  JSON saved  : {json_path}")

    out_img = annotate_image(args.images[-1], result, args.output)
    if out_img:
        print(f"🖼   Image saved : {out_img}")


def main():
    parser = argparse.ArgumentParser(
        description="Visual Diff Detector — powered by Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("images", nargs="*", help="Two or more image paths (before → after)")
    parser.add_argument("--api-key", "-k", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--output", "-o", default="annotated_diff.jpg", help="Output image path (default: annotated_diff.jpg)")
    args = parser.parse_args()

    if len(args.images) >= 2:
        run_cli(args)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  CLI: python claude_change_detect.py before.jpg after.jpg [extra.jpg ...] --api-key sk-ant-...")


if __name__ == "__main__":
    main()