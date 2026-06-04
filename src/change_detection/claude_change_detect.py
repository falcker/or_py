#!/usr/bin/env python3
"""
Visual Diff Detector — powered by Claude
=========================================
Detects visual differences between a target image and one or more reference
and/or labeled-example images using the Anthropic API.

Usage (new explicit mode):
    python claude_change_detect.py \\
        --ref baseline_1.jpg --ref baseline_2.jpg \\
        --example water_ref.png:water_pooling \\
        target.jpg

Usage (legacy positional mode, kept for back-compat):
    python claude_change_detect.py before.jpg after.jpg
    python claude_change_detect.py before.jpg mid.jpg after.jpg

Prompt control:
    --prompt leak_focus            # built-in name (see prompts.BUILTIN_PROMPTS)
    --prompt ./my_prompt.txt       # path to a text file
    --prompt -                     # read prompt from stdin
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

from change_detection.prompts import BUILTIN_PROMPTS

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    print("Warning: ANTHROPIC_API_KEY not set. CLI mode will not work without it.")


DEFAULT_PROMPT_NAME = "leak_focus"


# ─────────────────────────────────────────────
#  Image helpers
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


ANOMALY_COLORS = [
    (255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0),
    (175, 82, 222), (255, 45, 85), (88, 86, 214), (90, 200, 250),
]


def build_message_content(refs: list[str], examples: list[dict], current: str, prompt: str) -> list[dict]:
    """Build the message content list (images + text) that goes to the API.

    `examples` is a list of dicts: {"path": str, "type": str, "bbox": Any}.
    The image order matches build_layout_block: refs, then examples, then current.
    """
    content: list[dict] = []
    for p in refs:
        content.append(build_image_block(p))
    for ex in examples:
        content.append(build_image_block(ex["path"]))
    content.append(build_image_block(current))
    content.append({"type": "text", "text": prompt})
    return content


def render_content_as_text(content: list[dict]) -> str:
    """Render a message-content list into a flat text dump for logging.

    Image blocks are reduced to placeholders so the log stays human-readable
    and small (we don't dump base64).
    """
    lines: list[str] = []
    img_idx = 0
    for block in content:
        if block.get("type") == "image":
            img_idx += 1
            mime = block.get("source", {}).get("media_type", "image/?")
            lines.append(f"[IMAGE #{img_idx} ({mime})]")
        elif block.get("type") == "text":
            lines.append("")
            lines.append(block.get("text", ""))
        else:
            lines.append(f"[UNKNOWN BLOCK: {block.get('type')}]")
    return "\n".join(lines)


def normalize_result(parsed) -> dict:
    """Coerce the model's parsed JSON into {'anomalies': [...]} shape.

    Accepts both the new multi-anomaly schema and the legacy single-bounding-box
    schema, so prompts of either generation still work.
    """
    if isinstance(parsed, list):
        return {"anomalies": parsed}
    if isinstance(parsed, dict):
        if isinstance(parsed.get("anomalies"), list):
            return {"anomalies": parsed["anomalies"]}
        if "bounding_box" in parsed:
            bb = parsed.get("bounding_box", {}) or {}
            if all(bb.get(k, 0) == 0 for k in ("x", "y", "width", "height")):
                return {"anomalies": []}
            return {"anomalies": [{
                "description": parsed.get("description", ""),
                "bounding_box": bb,
            }]}
    return {"anomalies": []}


# ─────────────────────────────────────────────
#  Prompt assembly
# ─────────────────────────────────────────────

def load_prompt(source: str) -> str:
    """Resolve a --prompt argument to the actual prompt text.

    Accepts:
      - "-"                         → read from stdin
      - a path to an existing file  → read file contents
      - a key in BUILTIN_PROMPTS    → use the registered prompt
    """
    if source == "-":
        return sys.stdin.read()

    p = Path(source)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")

    if source in BUILTIN_PROMPTS:
        return BUILTIN_PROMPTS[source]

    raise ValueError(
        f"--prompt '{source}' is neither a file path nor a known prompt name. "
        f"Known names: {', '.join(sorted(BUILTIN_PROMPTS))}"
    )


def build_composite_image(
    ref_paths: list[str],
    target_path: str,
    output_path: str,
    *,
    label_refs: str = "REFERENCE",
    label_target: str = "TARGET (inspect this)",
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Build a single composite image: refs stacked vertically on the LEFT,
    target on the RIGHT, with colored label bands so the model can tell them
    apart. Used when an upstream tool (e.g. Roboflow) only accepts a single
    image per Claude API call.

    Returns ``(output_path, target_offset_in_composite, target_dims)`` where
    target_offset = (x, y) of the target panel's top-left corner in the
    composite, and target_dims = (width, height) of the target panel.
    """
    from PIL import Image, ImageDraw, ImageFont

    refs = [Image.open(p).convert("RGB") for p in ref_paths]
    tgt  = Image.open(target_path).convert("RGB")

    panel_h = tgt.height
    tgt_w   = tgt.width

    # Scale each ref to fit a vertical slot of height panel_h / n_refs
    n_refs = max(1, len(refs))
    slot_h = panel_h // n_refs if n_refs > 1 else panel_h
    refs_scaled = []
    for r in refs:
        scale = slot_h / r.height
        new_w = max(1, int(r.width * scale))
        refs_scaled.append(r.resize((new_w, slot_h), Image.LANCZOS))
    ref_panel_w = max((r.width for r in refs_scaled), default=tgt_w // 2)

    # Label band — proportional but bounded
    label_band_h = max(30, min(90, panel_h // 15))
    font_size    = max(16, label_band_h - 14)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    composite_w = ref_panel_w + tgt_w
    composite_h = panel_h + label_band_h

    img = Image.new("RGB", (composite_w, composite_h), (40, 40, 40))
    draw = ImageDraw.Draw(img)

    # Label bands (ref = grey, target = red so it's unmistakable)
    draw.rectangle([0, 0, ref_panel_w, label_band_h], fill=(70, 70, 78))
    draw.rectangle([ref_panel_w, 0, composite_w, label_band_h], fill=(190, 50, 50))
    pad = max(8, label_band_h // 6)
    text_y = (label_band_h - font_size) // 2
    draw.text((pad, text_y), label_refs, fill=(255, 255, 255), font=font)
    draw.text((ref_panel_w + pad, text_y), label_target, fill=(255, 255, 255), font=font)

    # Paste refs (centered within their slot)
    y = label_band_h
    for r in refs_scaled:
        x = (ref_panel_w - r.width) // 2
        img.paste(r, (x, y))
        y += r.height
    # Paste target
    img.paste(tgt, (ref_panel_w, label_band_h))

    # Thin divider line between ref panel and target panel
    draw.line([(ref_panel_w, 0), (ref_panel_w, composite_h)], fill=(20, 20, 20), width=2)

    img.save(output_path, quality=92)
    target_offset = (ref_panel_w, label_band_h)
    target_dims   = (tgt_w, panel_h)
    return output_path, target_offset, target_dims


def build_layout_block_composite(
    n_refs: int, target_w: int, target_h: int, n_examples: int = 0
) -> str:
    """Layout preamble for composite-image mode (single image to the model)."""
    refs_word = "REFERENCE panel" if n_refs == 1 else f"{n_refs} REFERENCE panels"
    ex_note = ""
    if n_examples:
        ex_note = (f"\n  - {n_examples} EXAMPLE panel(s) calibrate anomaly categories; "
                   f"do NOT report changes found inside an example panel.")
    return (
        "Image input:\n"
        "  ONE composite image. Layout: " + refs_word + " on the LEFT (grey label band), "
        "the TARGET panel on the RIGHT (red label band labeled 'TARGET (inspect this)').\n"
        "  - The REFERENCE panel(s) show the baseline/normal state. Anything visible "
        "there is NOT an anomaly, even if it looks like wetness or staining."
        + ex_note + "\n"
        "  - Inspect the TARGET panel ONLY. Never report a region inside a reference panel.\n"
        f"  - The TARGET panel is {target_w} px wide × {target_h} px tall.\n"
        "  - Bounding box coordinates MUST be expressed in the TARGET panel's own "
        "coordinate space (origin (0,0) is the top-left corner of the TARGET panel — "
        "NOT the composite image). Do not include the reference panel in your coordinates."
    )


def build_layout_block(refs: list[str], examples: list[tuple[str, str]], target: str) -> str:
    """Generate the 'Image inputs' description that precedes the user prompt.

    This is the single source of truth for how each image position maps to its
    role. The user-supplied prompt never has to describe ordering itself.
    """
    lines = ["Image inputs (in order):"]
    idx = 1

    for _ in refs:
        lines.append(f"  {idx}. REFERENCE — a baseline/normal-state image. No anomaly is present here.")
        idx += 1

    for _, label in examples:
        lines.append(
            f"  {idx}. EXAMPLE — illustrates the anomaly type '{label}'. "
            f"Use it to calibrate what '{label}' looks like; do NOT report it as a change."
        )
        idx += 1

    target_line = f"  {idx}. TARGET — the image to inspect. Report changes in this image only."
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(target) as _img:
            tw, th = _img.size
        target_line += (
            f" Dimensions: {tw}×{th} px. "
            f"All bounding-box coordinates must be integers within "
            f"x ∈ [0, {tw - 1}], y ∈ [0, {th - 1}]."
        )
    except Exception:
        pass
    lines.append(target_line)

    if examples:
        labels = sorted({label for _, label in examples})
        lines.append("")
        lines.append("Anomaly categories illustrated by the examples: " + ", ".join(labels) + ".")

    return "\n".join(lines)


def assemble_prompt(
    base_prompt: str,
    refs: list[str],
    examples: list[tuple[str, str]],
    target: str,
    *,
    composite_dims: tuple[int, int] | None = None,
) -> str:
    """Prepend the layout description to the user prompt.

    If ``composite_dims`` is provided, use the single-image composite preamble
    (and ignore positional layout for refs/target). Otherwise use the standard
    multi-image preamble.
    """
    if composite_dims is not None:
        layout = build_layout_block_composite(
            n_refs=len(refs), target_w=composite_dims[0], target_h=composite_dims[1],
            n_examples=len(examples),
        )
    else:
        layout = build_layout_block(refs, examples, target)
    return f"{layout}\n\n{base_prompt.strip()}\n"


# ─────────────────────────────────────────────
#  Claude API
# ─────────────────────────────────────────────

def call_claude(image_paths: list[str], api_key: str, prompt: str) -> dict:
    """Send images to Claude and return the parsed JSON result."""
    if len(image_paths) < 1:
        raise ValueError("At least one image is required.")

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

def annotate_image(img_path: str, anomalies: list[dict], output_path: str,
                   bbox_offset: tuple[int, int] = (0, 0)) -> str:
    """Draw a bounding box per anomaly on the target image and save to output_path.

    ``bbox_offset`` is added to every (x, y) before drawing — used in composite
    mode where the model reports bboxes in TARGET-panel coordinates but we
    annotate the composite (so the offset is the target panel's top-left).
    """
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
    off_x, off_y = bbox_offset
    for idx, anomaly in enumerate(anomalies):
        bb = anomaly.get("bounding_box", {})
        x = bb.get("x", 0) + off_x
        y = bb.get("y", 0) + off_y
        w = bb.get("width", 0)
        h = bb.get("height", 0)

        if x == 0 and y == 0 and w == 0 and h == 0:
            print(f"⚠  Anomaly {idx + 1} has no bounding box — skipping.")
            continue

        if x >= img.width or y >= img.height or x + w <= 0 or y + h <= 0:
            print(f"⚠  Anomaly {idx + 1} bounding box ({x},{y},{w},{h}) is entirely outside "
                  f"image bounds ({img.width}×{img.height}) — skipping.")
            continue

        # Clamp to image bounds
        x2 = min(x + w, img.width)
        y2 = min(y + h, img.height)
        x = max(0, x)
        y = max(0, y)
        w = x2 - x
        h = y2 - y

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


def copy_inputs_to(
    target_dir: str | Path,
    ref_paths: list[str],
    examples: list[dict],
    current_path: str,
) -> dict:
    """Copy refs, examples, and the current image into <target_dir>/inputs/ using
    canonical names. Returns a dict with the paths (relative to target_dir).

    Idempotent: if a destination file already exists with the same size as the
    source, the copy is skipped. Names match what create_run_folder would have
    produced, so an externally-shared inputs/ folder is layout-compatible.
    """
    import shutil

    target_dir = Path(target_dir)
    inputs_dir = target_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    def _copy(src: Path, dest: Path) -> None:
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            return
        shutil.copy2(src, dest)

    saved_refs: list[str] = []
    saved_examples: list[dict] = []

    for idx, p in enumerate(ref_paths, start=1):
        src = Path(p)
        dest = inputs_dir / f"ref_{idx:02d}{src.suffix}"
        _copy(src, dest)
        saved_refs.append(str(dest.relative_to(target_dir)))

    for idx, ex in enumerate(examples, start=1):
        src = Path(ex["path"])
        dest_name = f"example_{idx:02d}_{_safe_filename(ex['type'])}{src.suffix}"
        dest = inputs_dir / dest_name
        _copy(src, dest)
        saved_examples.append({
            "path": str(dest.relative_to(target_dir)),
            "type": ex["type"],
            "bbox": ex["bbox"],
        })

    src_cur = Path(current_path)
    dest_cur = inputs_dir / f"current{src_cur.suffix}"
    _copy(src_cur, dest_cur)
    saved_current = str(dest_cur.relative_to(target_dir))

    return {
        "refs": saved_refs,
        "examples": saved_examples,
        "current": saved_current,
    }


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
    folder_name: str | None = None,
    external_inputs_dir: str | Path | None = None,
) -> str:
    """
    Create a timestamped run folder and save all artifacts.
    Returns the path of the created run folder.
    """
    from datetime import datetime
    import shutil
    import hashlib

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if folder_name is None:
        folder_name = timestamp
        if label:
            folder_name += f"_{_safe_filename(label)}"
    else:
        folder_name = _safe_filename(folder_name)

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

    # 4. Materialize input copies and the annotated output
    saved_refs: list[str] = []
    saved_examples: list[dict] = []
    saved_current: str | None = None
    saved_annotated: str | None = None
    shared_inputs_dir_rel: str | None = None

    if external_inputs_dir is not None:
        # Inputs live in a folder shared across sibling runs. Don't copy them
        # again — just record relative paths so consumers can find them.
        ext_dir = Path(external_inputs_dir).resolve()
        rel_to_run = os.path.relpath(ext_dir, start=run_dir.resolve())
        shared_inputs_dir_rel = rel_to_run

        for idx, p in enumerate(ref_paths, start=1):
            name = f"ref_{idx:02d}{Path(p).suffix}"
            saved_refs.append(f"{rel_to_run}/{name}".replace("\\", "/"))
        for idx, ex in enumerate(examples, start=1):
            name = f"example_{idx:02d}_{_safe_filename(ex['type'])}{Path(ex['path']).suffix}"
            saved_examples.append({
                "path": f"{rel_to_run}/{name}".replace("\\", "/"),
                "type": ex["type"],
                "bbox": ex["bbox"],
            })
        name = f"current{Path(current_path).suffix}"
        saved_current = f"{rel_to_run}/{name}".replace("\\", "/")
    elif copy_inputs:
        copied = copy_inputs_to(run_dir, ref_paths, examples, current_path)
        saved_refs = copied["refs"]
        saved_examples = copied["examples"]
        saved_current = copied["current"]

    if annotated_image_path and Path(annotated_image_path).exists():
        src_ann = Path(annotated_image_path).resolve()
        dest = (run_dir / f"annotated{src_ann.suffix}").resolve()
        if src_ann != dest:
            shutil.copy2(src_ann, dest)
        saved_annotated = str(dest.relative_to(run_dir.resolve()))

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
            "copies": {
                "refs": saved_refs,
                "examples": saved_examples,
                "current": saved_current,
                "annotated": saved_annotated,
            } if (copy_inputs or external_inputs_dir is not None) else None,
            "shared_inputs_dir": shared_inputs_dir_rel,
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

def parse_example(spec: str) -> tuple[str, str]:
    """Parse a 'path:label' argument. Handles Windows drive letters (C:\\...)."""
    sep = spec.rfind(":")
    if sep == -1 or sep <= 1:
        raise argparse.ArgumentTypeError(
            f"--example must be PATH:LABEL (got '{spec}')"
        )
    path, label = spec[:sep], spec[sep + 1 :]
    if not path or not label:
        raise argparse.ArgumentTypeError(
            f"--example must be PATH:LABEL with both parts non-empty (got '{spec}')"
        )
    return path, label


def resolve_inputs(args) -> tuple[list[str], list[tuple[str, str]], str, list[str]]:
    """Return (refs, examples, target, all_image_paths_in_send_order)."""
    refs: list[str] = list(args.ref or [])
    examples: list[tuple[str, str]] = list(args.example or [])

    if refs or examples:
        if len(args.images) != 1:
            sys.exit(
                "Error: when using --ref/--example, provide exactly one positional "
                f"target image (got {len(args.images)})."
            )
        target = args.images[0]
    else:
        # Legacy mode: all positional, first N-1 are refs, last is target.
        if len(args.images) < 2:
            sys.exit("Error: provide at least two images, or use --ref/--example with a target.")
        refs = args.images[:-1]
        target = args.images[-1]

    all_paths = refs + [p for p, _ in examples] + [target]
    missing = [p for p in all_paths if not Path(p).exists()]
    if missing:
        sys.exit("Error: file(s) not found:\n  " + "\n  ".join(missing))

    return refs, examples, target, all_paths


def run_cli(args):
    key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key and not args.dry_run:
        sys.exit("Error: provide --api-key or set ANTHROPIC_API_KEY.")

    refs, examples, target, all_paths = resolve_inputs(args)

    base_prompt = load_prompt(args.prompt)

    composite_path: str | None = None
    target_offset: tuple[int, int] = (0, 0)
    if args.merge_input:
        if examples:
            print("⚠  --merge-input ignores --example panels; only refs + target are composed.")
        composite_path = str(Path(args.output).with_name(Path(args.output).stem + "_composite.jpg"))
        _, target_offset, target_dims = build_composite_image(refs, target, composite_path)
        final_prompt = assemble_prompt(base_prompt, refs, examples, target,
                                       composite_dims=target_dims)
        api_images = [composite_path]
    else:
        final_prompt = assemble_prompt(base_prompt, refs, examples, target)
        api_images = all_paths

    if args.show_prompt:
        print("── Final prompt ────────────────────────")
        print(final_prompt)
        print("────────────────────────────────────────\n")

    if args.dry_run:
        if args.merge_input:
            print(f"Dry run — would send 1 composite image ({composite_path}):")
            print(f"  built from {len(refs)} ref(s) + 1 target")
        else:
            print(f"Dry run — would send {len(all_paths)} image(s):")
            for i, p in enumerate(all_paths, 1):
                print(f"  {i}. {p}")
        return

    mode_desc = (f"1 composite image (merged from {len(refs)} ref + 1 target)"
                 if args.merge_input
                 else f"{len(all_paths)} image(s) ({len(refs)} ref, {len(examples)} example, 1 target)")
    print(f"🔍  Sending {mode_desc} to Claude...")
    result, usage = call_claude(api_images, key, final_prompt)
    anomalies = result.get("anomalies", [])

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
            bb = anomaly.get("bounding_box", {}) or {}
            print(f"      Box       : x={bb.get('x')} y={bb.get('y')} "
                  f"w={bb.get('width')} h={bb.get('height')}")
            print()

    # Save full result as JSON (including token usage)
    json_path = str(Path(args.output).with_suffix(".json"))
    with open(json_path, "w") as f:
        json.dump({"anomalies": anomalies, "usage": usage}, f, indent=2)
    print(f"\n✅  JSON saved  : {json_path}")

    # In merge-input mode, annotate the composite (with target-panel-coord
    # offset). Otherwise annotate the original target image.
    if args.merge_input and composite_path:
        out_img = annotate_image(composite_path, anomalies, args.output,
                                 bbox_offset=target_offset)
    else:
        out_img = annotate_image(target, anomalies, args.output)
    if out_img:
        print(f"🖼   Image saved : {out_img}")

    # Log the run (unless --no-log was passed)
    if not args.no_log:
        examples_for_log = [
            {"path": p, "type": label, "bbox": None}
            for p, label in examples
        ]
        run_dir = create_run_folder(
            runs_dir=args.runs_dir,
            label=args.label,
            ref_paths=refs,
            examples=examples_for_log,
            current_path=target,
            prompt=final_prompt,
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
    parser.add_argument("images", nargs="*",
                        help="Target image (when using --ref/--example) "
                             "or legacy ordered list of images (before...after).")
    parser.add_argument("--ref", action="append", metavar="PATH",
                        help="Reference/baseline image. Repeat for multiple.")
    parser.add_argument("--example", action="append", type=parse_example, metavar="PATH:LABEL",
                        help="Labeled example of an anomaly (e.g. 'water_ref.png:water_pooling'). Repeat for multiple.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_NAME,
                        help=f"Prompt source: built-in name ({', '.join(sorted(BUILTIN_PROMPTS))}), "
                             f"file path, or '-' for stdin. Default: {DEFAULT_PROMPT_NAME}.")
    parser.add_argument("--show-prompt", action="store_true",
                        help="Print the fully assembled prompt before sending.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve inputs and prompt, but do not call the API.")
    parser.add_argument("--merge-input", action="store_true",
                        help="Compose refs + target into a single labeled image and "
                             "send only that to the API. Useful when an upstream tool "
                             "(e.g. Roboflow) restricts uploads to one image. The "
                             "composite is saved alongside --output; bbox coordinates "
                             "are reported in TARGET-panel space.")
    parser.add_argument("--api-key", "-k", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--output", "-o", default="annotated_diff.jpg",
                        help="Output annotated image path (default: annotated_diff.jpg)")
    parser.add_argument("--runs-dir", default="dev/runs",
                        help="Directory for per-run logs (default: dev/runs)")
    parser.add_argument("--label", default=None,
                        help="Optional label appended to the run folder name")
    parser.add_argument("--no-log", action="store_true",
                        help="Do not write a per-run log folder.")
    parser.add_argument("--no-copy-inputs", action="store_true",
                        help="Do not copy input images into the run folder.")
    args = parser.parse_args()

    has_inputs = args.ref or args.example or len(args.images) >= 2
    if not has_inputs:
        parser.print_help()
        return

    run_cli(args)


if __name__ == "__main__":
    main()
