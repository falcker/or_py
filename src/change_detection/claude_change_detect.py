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

    lines.append(f"  {idx}. TARGET — the image to inspect. Report changes in this image only.")

    if examples:
        labels = sorted({label for _, label in examples})
        lines.append("")
        lines.append("Anomaly categories illustrated by the examples: " + ", ".join(labels) + ".")

    return "\n".join(lines)


def assemble_prompt(base_prompt: str, refs: list[str], examples: list[tuple[str, str]], target: str) -> str:
    layout = build_layout_block(refs, examples, target)
    return f"{layout}\n\n{base_prompt.strip()}\n"


# ─────────────────────────────────────────────
#  Claude API
# ─────────────────────────────────────────────

def call_claude(image_paths: list[str], api_key: str, prompt: str) -> dict:
    """Send images to Claude and return the parsed JSON result."""
    if len(image_paths) < 2:
        raise ValueError("At least two images are required (one reference/example + one target).")

    content = [build_image_block(p) for p in image_paths] + [{"type": "text", "text": prompt}]

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

def annotate_image(img_path: str, result: dict, output_path: str) -> str:
    """Draw bounding box on the target image and save to output_path."""
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

    x, y, w, h = bb["x"], bb["y"], bb["width"], bb["height"]
    if x == 0 and y == 0 and w == 0 and h == 0:
        print("⚠  No bounding box detected — skipping annotation.")
        return ""
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
    if not key:
        sys.exit("Error: provide --api-key or set ANTHROPIC_API_KEY.")

    refs, examples, target, all_paths = resolve_inputs(args)

    base_prompt = load_prompt(args.prompt)
    final_prompt = assemble_prompt(base_prompt, refs, examples, target)

    if args.show_prompt:
        print("── Final prompt ────────────────────────")
        print(final_prompt)
        print("────────────────────────────────────────\n")

    if args.dry_run:
        print(f"Dry run — would send {len(all_paths)} image(s):")
        for i, p in enumerate(all_paths, 1):
            print(f"  {i}. {p}")
        return

    print(f"🔍  Sending {len(all_paths)} image(s) to Claude "
          f"({len(refs)} ref, {len(examples)} example, 1 target)...")
    result = call_claude(all_paths, key, final_prompt)

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
        json.dump({"bounding_box": bb, "description": result.get("description")}, f, indent=2)
    print(f"\n✅  JSON saved  : {json_path}")

    out_img = annotate_image(target, result, args.output)
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
    parser.add_argument("--api-key", "-k", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--output", "-o", default="annotated_diff.jpg",
                        help="Output annotated image path (default: annotated_diff.jpg)")
    args = parser.parse_args()

    has_inputs = args.ref or args.example or len(args.images) >= 2
    if not has_inputs:
        parser.print_help()
        return

    run_cli(args)


if __name__ == "__main__":
    main()
