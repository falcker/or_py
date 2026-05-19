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

from change_detection.prompts import COMPARISON_SYSTEM_PROMPT_LEAK_FOCUS_THREE_IMAGES_5_0

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


def normalize_result(parsed) -> list[dict]:
    """
    Normalize Claude's response into a consistent list-of-anomalies format.
    Handles all shapes our prompts might return:
      - List of anomaly dicts (new schema)
      - Single anomaly dict (old schema)
      - Empty list (no anomalies found)
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # Old single-anomaly format — wrap in list
        if "bounding_box" in parsed or "description" in parsed:
            return [parsed]
        # Possibly wrapped: {"anomalies": [...]} or {"differences": [...]}
        for key in ("anomalies", "differences", "results"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
    return []


# ─────────────────────────────────────────────
#  Claude API
# ─────────────────────────────────────────────

def call_claude(image_paths: list[str], api_key: str, prompt: str) -> list[dict]:
    """Send two or more images to Claude and return a list of anomaly dicts."""
    if len(image_paths) < 2:
        raise ValueError("At least two images are required.")

    content = [build_image_block(p) for p in image_paths] + [{"type": "text", "text": prompt}]

    payload = {
        "model": "claude-opus-4-7",
        "max_tokens": 10000,
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
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse Claude response as JSON:\n{text}") from e

    return normalize_result(parsed)


# ─────────────────────────────────────────────
#  Image annotation (Pillow)
# ─────────────────────────────────────────────

# Color palette for multiple anomalies (cycles if more than 6)
ANOMALY_COLORS = [
    (255, 59, 48),    # red
    (255, 149, 0),    # orange
    (255, 204, 0),    # yellow
    (52, 199, 89),    # green
    (0, 122, 255),    # blue
    (175, 82, 222),   # purple
]


def annotate_image(img_path: str, anomalies: list[dict], output_path: str) -> str:
    """Draw bounding boxes for all anomalies on the last image and save."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("⚠  Pillow not installed — skipping annotation. Install with: pip install pillow")
        return ""

    if not anomalies:
        print("⚠  No anomalies to annotate.")
        return ""

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    lw = max(4, img.width // 400)
    font_size = max(14, img.width // 120)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    drawn_any = False
    for idx, anomaly in enumerate(anomalies):
        bb = anomaly.get("bounding_box", {})
        x = bb.get("x", 0)
        y = bb.get("y", 0)
        w = bb.get("width", 0)
        h = bb.get("height", 0)

        if x == 0 and y == 0 and w == 0 and h == 0:
            print(f"⚠  Anomaly {idx + 1} has no bounding box — skipping.")
            continue

        color = ANOMALY_COLORS[idx % len(ANOMALY_COLORS)]
        color_fill = (*color, 40)
        color_outline = (*color, 255)
        color_label_bg = (*color, 230)

        draw.rectangle([x, y, x + w, y + h], fill=color_fill)
        draw.rectangle([x, y, x + w, y + h], outline=color_outline, width=lw)

        # Build label with type/severity/confidence when available
        anomaly_type = anomaly.get("anomaly_type", "anomaly")
        severity = anomaly.get("severity", "")
        confidence = anomaly.get("confidence")
        label_parts = [f"{idx + 1}. {anomaly_type}"]
        if severity:
            label_parts.append(severity)
        if isinstance(confidence, (int, float)):
            label_parts.append(f"{confidence:.0%}")
        label = " | ".join(label_parts)

        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        pad = 6
        lx = x
        ly = max(0, y - th - pad * 2)
        draw.rectangle([lx, ly, lx + tw + pad * 2, ly + th + pad * 2], fill=color_label_bg)
        draw.text((lx + pad, ly + pad), label, fill=(255, 255, 255, 255), font=font)
        drawn_any = True

    if not drawn_any:
        print("⚠  No valid bounding boxes to draw.")
        return ""

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
    anomalies = call_claude(args.images, key, COMPARISON_SYSTEM_PROMPT_LEAK_FOCUS_THREE_IMAGES_5_0)

    print("\n── Result ──────────────────────────────")
    if not anomalies:
        print("✅  No anomalies detected.")
    else:
        print(f"⚠   {len(anomalies)} anomaly/anomalies detected:\n")
        for idx, anomaly in enumerate(anomalies, start=1):
            print(f"  [{idx}] {anomaly.get('description', 'N/A')}")
            if anomaly.get("anomaly_type"):
                print(f"      Type      : {anomaly['anomaly_type']}")
            if anomaly.get("severity"):
                print(f"      Severity  : {anomaly['severity']}")
            conf = anomaly.get("confidence")
            if isinstance(conf, (int, float)):
                print(f"      Confidence: {conf:.0%}")
            bb = anomaly.get("bounding_box", {})
            print(f"      Box       : x={bb.get('x')} y={bb.get('y')} "
                  f"w={bb.get('width')} h={bb.get('height')}")
            print()

    # Save full result as JSON
    json_path = str(Path(args.output).with_suffix(".json"))
    with open(json_path, "w") as f:
        json.dump({"anomalies": anomalies}, f, indent=2)
    print(f"✅  JSON saved  : {json_path}")

    # Annotate the last (current) image
    out_img = annotate_image(args.images[-1], anomalies, args.output)
    if out_img:
        print(f"🖼   Image saved : {out_img}")


def main():
    parser = argparse.ArgumentParser(
        description="Visual Diff Detector — powered by Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("images", nargs="*", help="Two or more image paths (references first, current last)")
    parser.add_argument("--api-key", "-k", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--output", "-o", default="annotated_diff.jpg", help="Output image path (default: annotated_diff.jpg)")
    args = parser.parse_args()

    if len(args.images) >= 2:
        run_cli(args)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  CLI: python claude_change_detect.py reference1.jpg reference2.jpg current.jpg --api-key sk-ant-...")


if __name__ == "__main__":
    main()