#!/usr/bin/env python3
"""One-shot bounding-box annotator.

Usage:
    python -m change_detection.annotate_one IMAGE.jpg ANOMALIES.json [OUTPUT.jpg]
    python -m change_detection.annotate_one IMAGE.jpg --stdin   < anomalies.json
    python -m change_detection.annotate_one IMAGE.jpg --inline '{"anomalies":[...]}'

The JSON may be either:
    {"anomalies": [ {description, anomaly_type, severity, confidence, bounding_box, ...}, ... ]}
or a bare list of anomaly objects.

Output goes to OUTPUT.jpg (default: <image_stem>_annotated.jpg next to the input).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from change_detection.claude_change_detect import annotate_image, normalize_result


def load_anomalies(args) -> list[dict]:
    if args.inline:
        raw = json.loads(args.inline)
    elif args.stdin:
        raw = json.loads(sys.stdin.read())
    elif args.json:
        raw = json.loads(Path(args.json).read_text(encoding="utf-8"))
    else:
        sys.exit("Provide ANOMALIES.json, --inline JSON, or --stdin.")
    norm = normalize_result(raw)
    return norm["anomalies"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw bounding boxes from an anomalies JSON onto an image.")
    ap.add_argument("image", help="Path to the image to annotate.")
    ap.add_argument("json", nargs="?", default=None,
                    help="Path to anomalies JSON (or use --stdin / --inline).")
    ap.add_argument("output", nargs="?", default=None,
                    help="Output path. Default: <image_stem>_annotated.jpg")
    ap.add_argument("--stdin", action="store_true",
                    help="Read anomalies JSON from stdin.")
    ap.add_argument("--inline", default=None,
                    help="Inline JSON string of anomalies.")
    ap.add_argument("--offset", default=None,
                    help="Optional 'x,y' to add to every bbox (e.g. when "
                         "annotating a composite where bboxes are in target-panel coords).")
    args = ap.parse_args()

    img = Path(args.image)
    if not img.exists():
        sys.exit(f"Image not found: {img}")
    out = Path(args.output) if args.output else img.with_name(img.stem + "_annotated.jpg")

    anomalies = load_anomalies(args)
    if not anomalies:
        sys.exit("No anomalies in the input — nothing to annotate.")

    offset = (0, 0)
    if args.offset:
        try:
            x_s, y_s = args.offset.split(",")
            offset = (int(x_s), int(y_s))
        except Exception:
            sys.exit("--offset must be 'x,y' integers (e.g. '845,805').")

    written = annotate_image(str(img), anomalies, str(out), bbox_offset=offset)
    if not written:
        sys.exit("Annotation failed (see warnings above).")
    print(f"Wrote {out}")
    for i, a in enumerate(anomalies, 1):
        bb = a.get("bounding_box", {})
        print(f"  [{i}] {a.get('anomaly_type','?')}  "
              f"conf={a.get('confidence','?')}  "
              f"bbox=(x={bb.get('x')}, y={bb.get('y')}, "
              f"w={bb.get('width')}, h={bb.get('height')})")


if __name__ == "__main__":
    main()
