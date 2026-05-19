#!/usr/bin/env python3
"""
Visual Diff Detector — powered by Claude
=========================================
Detects visual differences between reference images and a current image, with optional
few-shot examples of known anomaly types (water_pooling, rust, oil_stain, etc.).

Image roles:
  --ref path           Normal reference image (repeatable). Shows the site in its
                       expected state across lighting/weather variations.
  --example path:type[:x,y,w,h]
                       Few-shot anomaly example (repeatable). Type is required and
                       becomes the anomaly_type label. Optional bounding box tells
                       Claude exactly where the anomaly is in the example image.
  current              Last positional argument. The image to inspect.

Usage:
    # Plain reference + current
    python claude_change_detect.py --ref normal1.jpg --ref normal2.jpg current.jpg

    # With anomaly examples (few-shot)
    python claude_change_detect.py \\
        --ref normal1.jpg --ref normal2.jpg \\
        --example water_example.jpg:water_pooling:240,500,300,180 \\
        --example rust_example.jpg:corrosion \\
        --example oil_example.jpg:oil_stain:100,200,150,120 \\
        current.jpg

    # With API key and custom output
    python claude_change_detect.py --ref ref.jpg current.jpg \\
        --api-key sk-ant-... --output result.jpg
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


def build_message_content(
    ref_paths: list[str],
    examples: list[dict],
    current_path: str | None,
    prompt: str,
) -> list[dict]:
    """
    Build the full message content list (text + image blocks) sent to Claude.
    Used by both call_claude (real run) and export_prompt (preview).

    If current_path is None (preview/export mode), the current-image block is
    represented as a placeholder so the structure is still visible.
    """
    content: list[dict] = []

    # 1. Normal references
    if ref_paths:
        content.append({
            "type": "text",
            "text": (
                f"The following {len(ref_paths)} image(s) are NORMAL REFERENCE images "
                f"showing the equipment in its expected state across natural variation "
                f"(lighting, weather, wet/dry surfaces, seasonal changes)."
            ),
        })
        for p in ref_paths:
            content.append({"type": "image_placeholder", "path": p})

    # 2. Anomaly examples (few-shot)
    if examples:
        content.append({
            "type": "text",
            "text": (
                f"The following {len(examples)} image(s) are FEW-SHOT EXAMPLES of "
                f"confirmed anomalies at similar sites. Each is labeled with its "
                f"anomaly_type. Use these to calibrate what each anomaly type looks "
                f"like in practice — do NOT treat these as references for the current image."
            ),
        })
        for ex in examples:
            label = f"Anomaly example — type: {ex['type']}"
            if ex["bbox"]:
                bb = ex["bbox"]
                label += (
                    f". The anomaly is located at bounding box "
                    f"x={bb['x']}, y={bb['y']}, width={bb['width']}, height={bb['height']}."
                )
            content.append({"type": "text", "text": label})
            content.append({"type": "image_placeholder", "path": ex["path"]})

    # 3. Current image
    content.append({
        "type": "text",
        "text": "The following image is the CURRENT IMAGE to inspect. This is your main focus.",
    })
    if current_path:
        content.append({"type": "image_placeholder", "path": current_path})
    else:
        content.append({"type": "image_placeholder", "path": "<CURRENT_IMAGE>"})

    # 4. Instruction prompt
    content.append({"type": "text", "text": prompt})

    return content


def materialize_content(content: list[dict]) -> list[dict]:
    """Convert image_placeholder entries to real base64 image blocks for the API."""
    out = []
    for block in content:
        if block.get("type") == "image_placeholder":
            out.append(build_image_block(block["path"]))
        else:
            out.append(block)
    return out


def render_content_as_text(content: list[dict]) -> str:
    """Render the message content as a human-readable transcript."""
    lines = []
    lines.append("=" * 70)
    lines.append("FULL MESSAGE SENT TO CLAUDE")
    lines.append("=" * 70)
    lines.append("")
    for idx, block in enumerate(content, start=1):
        if block.get("type") == "text":
            lines.append(f"[Block {idx}] TEXT")
            lines.append("-" * 70)
            lines.append(block["text"])
            lines.append("")
        elif block.get("type") == "image_placeholder":
            lines.append(f"[Block {idx}] IMAGE: {block['path']}")
            lines.append("")
    return "\n".join(lines)


def export_prompt(
    prompt: str,
    output_path: str | None = None,
    ref_paths: list[str] | None = None,
    examples: list[dict] | None = None,
    current_path: str | None = None,
) -> str:
    """
    Export the active prompt to a file (or stdout if output_path is None).

    If ref_paths/examples are provided, exports the FULL message structure
    (text blocks + image placeholders). Otherwise exports just the prompt text.

    Returns the path written to, or an empty string for stdout-only.
    """
    if ref_paths or examples:
        content = build_message_content(
            ref_paths or [], examples or [], current_path, prompt
        )
        rendered = render_content_as_text(content)
    else:
        rendered = prompt

    if output_path is None:
        print(rendered)
        return ""

    out = Path(output_path)
    out.write_text(rendered, encoding="utf-8")
    return str(out)


def parse_example_spec(spec: str) -> dict:
    """
    Parse an --example argument of the form  path:type[:x,y,w,h]
    Returns: {"path": str, "type": str, "bbox": dict|None}

    Handles Windows paths with drive letters (e.g. C:\\folder\\file.png:type:bbox)
    by splitting from the RIGHT and reassembling the path.
    """
    # Split from the right: bbox (optional), type, then everything else is the path.
    # Detect whether the trailing segment is a bbox (4 comma-separated ints) or a type.
    parts = spec.rsplit(":", 2)

    # Case A: 3 segments — could be path:type:bbox  OR  C:\path:type  (Windows, 2 logical)
    # Case B: 2 segments — path:type  OR  C:\path (just a path, no type — invalid)
    # We disambiguate by checking whether the final segment looks like a bbox.

    def looks_like_bbox(s: str) -> bool:
        coords = s.split(",")
        if len(coords) != 4:
            return False
        try:
            [int(c.strip()) for c in coords]
            return True
        except ValueError:
            return False

    if len(parts) == 3 and looks_like_bbox(parts[2]):
        # path : type : bbox
        path, anomaly_type, bbox_str = parts[0], parts[1], parts[2]
    elif len(parts) >= 2:
        # path : type     (no bbox; if 3 segments, the first two rejoin as the path)
        # rsplit(":", 1) gives us the correct 2-way split for path:type
        path, anomaly_type = spec.rsplit(":", 1)
        bbox_str = None
    else:
        raise ValueError(
            f"Invalid --example spec: '{spec}'. "
            f"Expected format: path:type[:x,y,w,h]"
        )

    if not anomaly_type.strip():
        raise ValueError(f"Missing anomaly type in '{spec}'.")

    bbox = None
    if bbox_str:
        coords = bbox_str.split(",")
        try:
            x, y, w, h = (int(c.strip()) for c in coords)
        except ValueError as e:
            raise ValueError(f"Bbox coords must be integers in '{spec}'") from e
        bbox = {"x": x, "y": y, "width": w, "height": h}

    if not Path(path).exists():
        raise FileNotFoundError(f"Example image not found: {path}")

    return {"path": path, "type": anomaly_type, "bbox": bbox}


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

def call_claude(
    ref_paths: list[str],
    examples: list[dict],
    current_path: str,
    api_key: str,
    prompt: str,
) -> tuple[list[dict], dict]:
    """
    Send references + few-shot anomaly examples + current image to Claude.

    Returns:
        (anomalies, usage)
        - anomalies: list of anomaly dicts parsed from Claude's response
        - usage: dict with keys 'input_tokens', 'output_tokens',
                 'cache_creation_input_tokens', 'cache_read_input_tokens',
                 'total_tokens', 'model', 'stop_reason'
    """
    if not ref_paths:
        raise ValueError("At least one --ref image is required.")
    if not Path(current_path).exists():
        raise FileNotFoundError(f"Current image not found: {current_path}")

    content = materialize_content(
        build_message_content(ref_paths, examples, current_path, prompt)
    )

    payload = {
        "model": "claude-opus-4-5",
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

    # Extract usage information from the API response
    raw_usage = data.get("usage", {}) or {}
    input_tokens = raw_usage.get("input_tokens", 0)
    output_tokens = raw_usage.get("output_tokens", 0)
    cache_create = raw_usage.get("cache_creation_input_tokens", 0) or 0
    cache_read = raw_usage.get("cache_read_input_tokens", 0) or 0

    usage = {
        "model": data.get("model", payload["model"]),
        "stop_reason": data.get("stop_reason"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_create,
        "cache_read_input_tokens": cache_read,
        "total_tokens": input_tokens + output_tokens + cache_create + cache_read,
    }

    return normalize_result(parsed), usage


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
#  Run logging
# ─────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    """Replace characters that are problematic in filenames."""
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def create_run_folder(
    runs_dir: str,
    label: str | None,
    ref_paths: list[str],
    examples: list[dict],
    current_path: str,
    prompt: str,
    anomalies: list[dict],
    usage: dict,
    annotated_image_path: str | None,
    copy_inputs: bool = True,
) -> str:
    """
    Create a timestamped run folder and save all artifacts.
    Returns the path of the created run folder.
    """
    from datetime import datetime
    import shutil
    import hashlib

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = timestamp
    if label:
        folder_name += f"_{_safe_filename(label)}"

    run_dir = Path(runs_dir) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save the raw prompt
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    # 2. Save the full assembled message (text blocks + image placeholders)
    full_content = build_message_content(ref_paths, examples, current_path, prompt)
    (run_dir / "message.txt").write_text(
        render_content_as_text(full_content), encoding="utf-8"
    )

    # 3. Save just the anomalies (handy for diffing across runs)
    (run_dir / "anomalies.json").write_text(
        json.dumps(anomalies, indent=2), encoding="utf-8"
    )

    # 4. Copy input images and annotated output
    saved_refs: list[str] = []
    saved_examples: list[dict] = []
    saved_current: str | None = None
    saved_annotated: str | None = None

    if copy_inputs:
        inputs_dir = run_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)

        for idx, p in enumerate(ref_paths, start=1):
            src = Path(p)
            dest = inputs_dir / f"ref_{idx:02d}{src.suffix}"
            shutil.copy2(src, dest)
            saved_refs.append(str(dest.relative_to(run_dir)))

        for idx, ex in enumerate(examples, start=1):
            src = Path(ex["path"])
            dest_name = f"example_{idx:02d}_{_safe_filename(ex['type'])}{src.suffix}"
            dest = inputs_dir / dest_name
            shutil.copy2(src, dest)
            saved_examples.append({
                "path": str(dest.relative_to(run_dir)),
                "type": ex["type"],
                "bbox": ex["bbox"],
            })

        src_cur = Path(current_path)
        dest_cur = inputs_dir / f"current{src_cur.suffix}"
        shutil.copy2(src_cur, dest_cur)
        saved_current = str(dest_cur.relative_to(run_dir))

    if annotated_image_path and Path(annotated_image_path).exists():
        dest = run_dir / f"annotated{Path(annotated_image_path).suffix}"
        shutil.copy2(annotated_image_path, dest)
        saved_annotated = str(dest.relative_to(run_dir))

    # 5. Build the manifest
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    manifest = {
        "run_id": folder_name,
        "timestamp": timestamp,
        "label": label,
        "prompt_hash": prompt_hash,
        "inputs": {
            "refs": ref_paths,
            "examples": [
                {"path": ex["path"], "type": ex["type"], "bbox": ex["bbox"]}
                for ex in examples
            ],
            "current": current_path,
            "copies_in_run_folder": {
                "refs": saved_refs,
                "examples": saved_examples,
                "current": saved_current,
                "annotated": saved_annotated,
            } if copy_inputs else None,
        },
        "anomalies": anomalies,
        "usage": usage,
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 6. Update top-level pointer to the most recent run
    latest_path = Path(runs_dir) / "latest.json"
    latest_payload = {
        "run_id": folder_name,
        "run_dir": str(run_dir),
        "timestamp": timestamp,
        "label": label,
        "prompt_hash": prompt_hash,
        "model": usage.get("model"),
        "anomaly_count": len(anomalies),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    latest_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    return str(run_dir)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def run_cli(args):
    key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("Error: provide --api-key or set ANTHROPIC_API_KEY.")
        sys.exit(1)

    # Validate references
    if not args.ref:
        print("Error: at least one --ref image is required.")
        sys.exit(1)
    missing_refs = [p for p in args.ref if not Path(p).exists()]
    if missing_refs:
        print(f"Error: reference file(s) not found: {', '.join(missing_refs)}")
        sys.exit(1)

    # Validate current image
    if not args.current:
        print("Error: current image (last positional argument) is required.")
        sys.exit(1)
    if not Path(args.current).exists():
        print(f"Error: current image not found: {args.current}")
        sys.exit(1)

    # Parse example specs
    examples = []
    for spec in (args.example or []):
        try:
            examples.append(parse_example_spec(spec))
        except (ValueError, FileNotFoundError) as e:
            print(f"Error parsing --example: {e}")
            sys.exit(1)

    print(f"🔍  Sending to Claude:")
    print(f"    Normal refs : {len(args.ref)}  ({', '.join(args.ref)})")
    if examples:
        ex_summary = ", ".join(
            f"{ex['type']}{'+bbox' if ex['bbox'] else ''}" for ex in examples
        )
        print(f"    Examples    : {len(examples)} ({ex_summary})")
    else:
        print(f"    Examples    : 0")
    print(f"    Current     : {args.current}")

    anomalies, usage = call_claude(
        ref_paths=args.ref,
        examples=examples,
        current_path=args.current,
        api_key=key,
        prompt=COMPARISON_SYSTEM_PROMPT_LEAK_FOCUS_THREE_IMAGES_5_0,
    )

    print("\n── Token Usage ─────────────────────────")
    print(f"    Model              : {usage['model']}")
    print(f"    Stop reason        : {usage['stop_reason']}")
    print(f"    Input tokens       : {usage['input_tokens']:,}")
    print(f"    Output tokens      : {usage['output_tokens']:,}")
    if usage["cache_creation_input_tokens"]:
        print(f"    Cache creation     : {usage['cache_creation_input_tokens']:,}")
    if usage["cache_read_input_tokens"]:
        print(f"    Cache read         : {usage['cache_read_input_tokens']:,}")
    print(f"    TOTAL              : {usage['total_tokens']:,}")

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

    # Save full result as JSON (including token usage)
    json_path = str(Path(args.output).with_suffix(".json"))
    with open(json_path, "w") as f:
        json.dump({"anomalies": anomalies, "usage": usage}, f, indent=2)
    print(f"✅  JSON saved  : {json_path}")

    # Annotate the current image
    out_img = annotate_image(args.current, anomalies, args.output)
    if out_img:
        print(f"🖼   Image saved : {out_img}")

    # Log the run (unless --no-log was passed)
    if not args.no_log:
        run_dir = create_run_folder(
            runs_dir=args.runs_dir,
            label=args.label,
            ref_paths=args.ref,
            examples=examples,
            current_path=args.current,
            prompt=COMPARISON_SYSTEM_PROMPT_LEAK_FOCUS_THREE_IMAGES_5_0,
            anomalies=anomalies,
            usage=usage,
            annotated_image_path=out_img or None,
            copy_inputs=not args.no_copy_inputs,
        )
        print(f"📁  Run logged  : {run_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visual Diff Detector — powered by Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ref", "-r", action="append", default=[],
        help="Normal reference image (repeatable). Pass at least one.",
    )
    parser.add_argument(
        "--example", "-e", action="append", default=[],
        help="Few-shot anomaly example. Format: path:type[:x,y,w,h]  (repeatable). "
             "Type examples: water_pooling, oil_stain, corrosion, leak, discoloration.",
    )
    parser.add_argument(
        "current", nargs="?",
        help="The current image to inspect (positional, last argument).",
    )
    parser.add_argument("--api-key", "-k", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--output", "-o", default="annotated_diff.jpg",
                        help="Output image path (default: annotated_diff.jpg)")
    parser.add_argument(
        "--runs-dir", default="runs",
        help="Directory where per-run folders are created (default: runs).",
    )
    parser.add_argument(
        "--label", "-l", default=None,
        help="Optional short label appended to the run folder name (e.g. 'fewshot_v1').",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Disable run logging entirely. By default, every run creates a folder under --runs-dir.",
    )
    parser.add_argument(
        "--no-copy-inputs", action="store_true",
        help="Skip copying input images into the run folder (just record paths).",
    )
    parser.add_argument(
        "--export-prompt", metavar="PATH",
        help="Write the active prompt to PATH and exit (no API call).",
    )
    parser.add_argument(
        "--show-prompt", action="store_true",
        help="Print the active prompt to stdout and exit (no API call).",
    )
    args = parser.parse_args()

    # Short-circuit: export/show prompt without making any API call
    if args.show_prompt or args.export_prompt:
        # Parse refs and examples (if provided) so the export shows the full structure.
        # We do NOT require current image here, and we don't require files to exist
        # if no refs/examples were passed.
        parsed_examples = []
        for spec in (args.example or []):
            try:
                # Reuse parser but skip the existence check by catching it
                parsed_examples.append(parse_example_spec(spec))
            except FileNotFoundError:
                # Allow missing files when only previewing the prompt
                # Parse manually without existence check
                if spec.rsplit(":", 2)[-1].count(",") == 3:
                    path, atype, bbox_str = spec.rsplit(":", 2)
                    coords = [int(c) for c in bbox_str.split(",")]
                    parsed_examples.append({
                        "path": path, "type": atype,
                        "bbox": {"x": coords[0], "y": coords[1], "width": coords[2], "height": coords[3]},
                    })
                else:
                    path, atype = spec.rsplit(":", 1)
                    parsed_examples.append({"path": path, "type": atype, "bbox": None})
            except ValueError as e:
                print(f"Error parsing --example: {e}")
                sys.exit(1)

        if args.show_prompt:
            export_prompt(
                COMPARISON_SYSTEM_PROMPT_LEAK_FOCUS_THREE_IMAGES_5_0,
                ref_paths=args.ref or None,
                examples=parsed_examples or None,
                current_path=args.current,
            )
            return
        if args.export_prompt:
            path = export_prompt(
                COMPARISON_SYSTEM_PROMPT_LEAK_FOCUS_THREE_IMAGES_5_0,
                output_path=args.export_prompt,
                ref_paths=args.ref or None,
                examples=parsed_examples or None,
                current_path=args.current,
            )
            print(f"✅  Prompt exported to: {path}")
            return

    if args.ref and args.current:
        run_cli(args)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  Basic           : python claude_change_detect.py --ref normal.jpg current.jpg")
        print("  Multi-reference : python claude_change_detect.py --ref n1.jpg --ref n2.jpg current.jpg")
        print("  With examples   : python claude_change_detect.py \\")
        print("                      --ref n1.jpg --ref n2.jpg \\")
        print("                      --example water.jpg:water_pooling:240,500,300,180 \\")
        print("                      --example rust.jpg:corrosion \\")
        print("                      current.jpg")


if __name__ == "__main__":
    main()