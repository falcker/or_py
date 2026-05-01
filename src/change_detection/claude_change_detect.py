#!/usr/bin/env python3
"""
Visual Diff Detector — powered by Claude
=========================================
Detects visual differences between two images using the Anthropic API.

Usage (CLI):
    python image_diff_detector.py image1.jpg image2.jpg
    python image_diff_detector.py image1.jpg image2.jpg --api-key sk-ant-...
    python image_diff_detector.py image1.jpg image2.jpg --output result.jpg

Usage (web UI):
    python image_diff_detector.py --web
    python image_diff_detector.py --web --port 8080
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

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    print("Warning: ANTHROPIC_API_KEY not set. CLI mode will not work without it.")


# ─────────────────────────────────────────────
#  Claude API
# ─────────────────────────────────────────────

DEFAULT_PROMPT = """You are an expert visual inspection AI. Compare these two images carefully \
and identify any notable differences, anomalies, stains, damage, or changes between them.

Return ONLY a JSON object with this exact structure — no markdown, no explanation:
{
  "description": "brief description of the difference found",
  "bounding_box": {
    "x": <left edge in pixels>,
    "y": <top edge in pixels>,
    "width": <width in pixels>,
    "height": <height in pixels>
  }
}

The bounding box must tightly surround the area of difference in the second (after) image, \
using the full original image resolution."""


def encode_image(path: str) -> tuple[str, str]:
    """Return (base64_data, mime_type) for an image file."""
    mime, _ = mimetypes.guess_type(path)
    if mime not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        mime = "image/jpeg"
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode(), mime


def call_claude(img1_path: str, img2_path: str, api_key: str, prompt: str) -> dict:
    """Send both images to Claude and return the parsed JSON result."""
    b64_1, mime_1 = encode_image(img1_path)
    b64_2, mime_2 = encode_image(img2_path)

    payload = {
        "model": "claude-opus-4-5",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime_1, "data": b64_1}},
                    {"type": "image", "source": {"type": "base64", "media_type": mime_2, "data": b64_2}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
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
        body = e.read().decode()
        raise RuntimeError(f"API error {e.code}: {body}") from e

    text = "".join(b.get("text", "") for b in data.get("content", [])).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse Claude response as JSON:\n{text}") from e


# ─────────────────────────────────────────────
#  Image annotation (Pillow)
# ─────────────────────────────────────────────

def annotate_image(img2_path: str, result: dict, output_path: str) -> str:
    """Draw bounding box on image2 and save to output_path."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("⚠  Pillow not installed — skipping image annotation.")
        print("   Install with: pip install pillow")
        return ""

    bb = result["bounding_box"]
    img = Image.open(img2_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    x, y, w, h = bb["x"], bb["y"], bb["width"], bb["height"]
    lw = max(4, img.width // 400)

    # Filled highlight
    draw.rectangle([x, y, x + w, y + h], fill=(255, 59, 48, 40))
    # Red border
    draw.rectangle([x, y, x + w, y + h], outline=(255, 59, 48, 255), width=lw)

    # Label background
    label = "anomaly detected"
    font_size = max(14, img.width // 120)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox_text = draw.textbbox((0, 0), label, font=font)
    text_w = bbox_text[2] - bbox_text[0]
    text_h = bbox_text[3] - bbox_text[1]
    pad = 6
    label_x, label_y = x, max(0, y - text_h - pad * 2)
    draw.rectangle(
        [label_x, label_y, label_x + text_w + pad * 2, label_y + text_h + pad * 2],
        fill=(255, 59, 48, 230),
    )
    draw.text((label_x + pad, label_y + pad), label, fill=(255, 255, 255, 255), font=font)

    img.save(output_path, quality=92)
    return output_path


# ─────────────────────────────────────────────
#  CLI mode
# ─────────────────────────────────────────────

def run_cli(args):
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: provide --api-key or set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    img1, img2 = args.image1, args.image2
    for p in (img1, img2):
        if not Path(p).exists():
            print(f"Error: file not found: {p}")
            sys.exit(1)

    print(f"🔍  Sending images to Claude...")
    result = call_claude(img1, img2, api_key, DEFAULT_PROMPT)

    print("\n── Result ──────────────────────────────")
    print(f"📋  Description : {result.get('description', 'N/A')}")
    bb = result.get("bounding_box", {})
    print(f"📦  Bounding box: x={bb.get('x')} y={bb.get('y')} w={bb.get('width')} h={bb.get('height')}")

    # Save JSON
    json_path = args.output.replace(".jpg", ".json").replace(".jpeg", ".json").replace(".png", ".json")
    if not json_path.endswith(".json"):
        json_path += ".json"
    with open(json_path, "w") as f:
        json.dump({"bounding_box": bb}, f, indent=2)
    print(f"\n✅  JSON saved  : {json_path}")

    # Annotate image
    out_img = annotate_image(img2, result, args.output)
    if out_img:
        print(f"🖼   Image saved : {out_img}")


# ─────────────────────────────────────────────
#  Web UI mode (built-in HTTP server)
# ─────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visual Diff Detector</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'DM Sans', 'Helvetica Neue', sans-serif;
  background: #0d0d0f; color: #e8e6e0;
  min-height: 100vh; padding: 2rem;
}
header {
  display: flex; align-items: baseline; gap: 1rem;
  margin-bottom: 2rem; border-bottom: 0.5px solid #2a2a2e; padding-bottom: 1rem;
}
header h1 { font-size: 18px; font-weight: 500; }
header span { font-size: 12px; color: #555; font-family: monospace; }
.row { display: flex; gap: 8px; align-items: center; margin-bottom: 1.25rem; }
.row label { font-size: 12px; color: #555; white-space: nowrap; font-family: monospace; }
.row input[type=password], .row input[type=text] {
  flex: 1; background: #1a1a1e; border: 0.5px solid #2a2a2e; border-radius: 6px;
  padding: 8px 12px; font-size: 13px; font-family: monospace; color: #e8e6e0; outline: none;
}
.upload-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 1.25rem; }
.drop-zone {
  background: #1a1a1e; border: 0.5px solid #2a2a2e; border-radius: 8px;
  aspect-ratio: 4/3; display: flex; flex-direction: column;
  align-items: center; justify-content: center; cursor: pointer;
  position: relative; overflow: hidden; transition: border-color 0.15s;
}
.drop-zone:hover { border-color: #444; }
.drop-zone img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.drop-zone .tag {
  position: absolute; top: 8px; left: 8px;
  background: rgba(0,0,0,0.7); font-size: 10px; font-family: monospace;
  color: #888; padding: 3px 7px; border-radius: 4px; text-transform: uppercase; z-index: 2;
}
.drop-zone .lbl { font-size: 12px; color: #555; text-align: center; }
.drop-zone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; z-index: 3; }
.prompt-wrap { margin-bottom: 1.25rem; }
.prompt-wrap label { display: block; font-size: 11px; color: #555; font-family: monospace; margin-bottom: 6px; text-transform: uppercase; letter-spacing: .05em; }
textarea {
  width: 100%; background: #1a1a1e; border: 0.5px solid #2a2a2e; border-radius: 6px;
  padding: 10px 12px; font-size: 13px; color: #e8e6e0; resize: vertical; min-height: 90px;
  outline: none; line-height: 1.6; font-family: sans-serif;
}
.run-btn {
  width: 100%; padding: 11px; background: #e8e6e0; color: #0d0d0f;
  border: none; border-radius: 6px; font-size: 14px; font-weight: 500;
  cursor: pointer; margin-bottom: 1rem; transition: background .15s;
}
.run-btn:hover { background: #fff; }
.run-btn:disabled { background: #2a2a2e; color: #555; cursor: not-allowed; }
.status { font-size: 12px; font-family: monospace; min-height: 18px; margin-bottom: 1rem; }
.status.running { color: #888; } .status.done { color: #5a9e6f; } .status.error { color: #c05050; }
.results { display: none; gap: 12px; }
.results.visible { display: grid; grid-template-columns: 1fr 1fr; }
.panel { background: #1a1a1e; border: 0.5px solid #2a2a2e; border-radius: 8px; overflow: hidden; }
.panel-header {
  padding: 9px 14px; border-bottom: 0.5px solid #2a2a2e;
  font-size: 11px; font-family: monospace; color: #555;
  text-transform: uppercase; letter-spacing: .05em;
  display: flex; align-items: center; justify-content: space-between;
}
canvas { display: block; width: 100%; }
.json-body { padding: 14px; font-family: monospace; font-size: 12px; line-height: 1.7; color: #888; white-space: pre; }
.k { color: #6699cc; } .n { color: #f0b060; }
.copy-btn {
  background: transparent; border: 0.5px solid #333; border-radius: 4px;
  color: #555; font-size: 10px; font-family: monospace; padding: 2px 8px; cursor: pointer;
}
.copy-btn:hover { border-color: #555; color: #888; }
.desc-panel { grid-column: 1/-1; background: #1a1a1e; border: 0.5px solid #2a2a2e; border-radius: 8px; padding: 14px; font-size: 13px; color: #aaa; line-height: 1.7; }
.desc-panel strong { color: #e8e6e0; font-weight: 500; }
.dl-row { grid-column: 1/-1; display: flex; gap: 8px; }
.dl-btn {
  flex: 1; padding: 9px; background: transparent; border: 0.5px solid #2a2a2e;
  border-radius: 6px; color: #888; font-size: 13px; cursor: pointer; font-family: monospace;
  transition: border-color .15s, color .15s;
}
.dl-btn:hover { border-color: #555; color: #ccc; }
</style>
</head>
<body>
<header><h1>Visual Diff Detector</h1><span>claude-opus-4-5 · local python server</span></header>

<div class="row">
  <label>API Key</label>
  <input type="password" id="apiKey" placeholder="sk-ant-..." />
</div>

<div class="upload-grid">
  <div class="drop-zone" id="z1">
    <span class="tag">before</span>
    <span class="lbl" id="lbl1">click to upload</span>
    <input type="file" accept="image/*" onchange="load(1,this)" />
  </div>
  <div class="drop-zone" id="z2">
    <span class="tag">after</span>
    <span class="lbl" id="lbl2">click to upload</span>
    <input type="file" accept="image/*" onchange="load(2,this)" />
  </div>
</div>

<div class="prompt-wrap">
  <label>Prompt</label>
  <textarea id="prompt">You are an expert visual inspection AI. Compare these two images carefully and identify any notable differences, anomalies, stains, damage, or changes between them.

Return ONLY a JSON object with this exact structure — no markdown, no explanation:
{
  "description": "brief description of the difference found",
  "bounding_box": {
    "x": <left edge in pixels>,
    "y": <top edge in pixels>,
    "width": <width in pixels>,
    "height": <height in pixels>
  }
}

The bounding box must tightly surround the area of difference in the second (after) image, using the full original image resolution.</textarea>
</div>

<button class="run-btn" id="runBtn" onclick="run()" disabled>Detect differences</button>
<div class="status" id="status"></div>

<div class="results" id="results">
  <div class="panel">
    <div class="panel-header">annotated result</div>
    <canvas id="canvas"></canvas>
  </div>
  <div class="panel">
    <div class="panel-header"><span>bounding box json</span><button class="copy-btn" onclick="copyJ()">copy</button></div>
    <div class="json-body" id="jsonBody"></div>
  </div>
  <div class="desc-panel" id="descPanel"></div>
  <div class="dl-row">
    <button class="dl-btn" onclick="dlImg()">⬇ download annotated image</button>
    <button class="dl-btn" onclick="dlJSON()">⬇ download json</button>
  </div>
</div>

<script>
const S = { img1: null, img2: null, lastJ: null };

function load(n, input) {
  const file = input.files[0];
  if (!file) return;
  const r = new FileReader();
  r.onload = e => {
    const src = e.target.result;
    const zone = document.getElementById('z' + n);
    let img = zone.querySelector('img');
    if (!img) { img = document.createElement('img'); zone.appendChild(img); }
    img.src = src;
    document.getElementById('lbl' + n).style.display = 'none';
    S['img' + n] = { data: src.split(',')[1], mime: src.split(';')[0].split(':')[1] };
    upd();
  };
  r.readAsDataURL(file);
}

function upd() {
  document.getElementById('runBtn').disabled = !(S.img1 && S.img2 && document.getElementById('apiKey').value.trim());
}
document.getElementById('apiKey').addEventListener('input', upd);

async function run() {
  const key = document.getElementById('apiKey').value.trim();
  const prompt = document.getElementById('prompt').value;
  document.getElementById('runBtn').disabled = true;
  setStatus('running', 'Sending to Claude...');
  document.getElementById('results').classList.remove('visible');
  try {
    const resp = await fetch('/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: key, prompt, img1: S.img1, img2: S.img2 })
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    S.lastJ = data;
    setStatus('done', 'Detection complete');
    render(data);
  } catch (e) {
    setStatus('error', 'Error: ' + e.message);
  } finally {
    document.getElementById('runBtn').disabled = false;
  }
}

function setStatus(t, m) {
  const el = document.getElementById('status');
  el.className = 'status ' + t;
  el.textContent = m;
}

function render(data) {
  const bb = data.bounding_box;
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width; canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    const lw = Math.max(4, img.width / 400);
    ctx.strokeStyle = '#ff3b30'; ctx.lineWidth = lw;
    ctx.strokeRect(bb.x, bb.y, bb.width, bb.height);
    ctx.fillStyle = 'rgba(255,59,48,0.12)';
    ctx.fillRect(bb.x + lw, bb.y + lw, bb.width - lw*2, bb.height - lw*2);
    const fs = Math.max(14, img.width / 120);
    ctx.font = `${fs}px monospace`;
    const tw = ctx.measureText('anomaly detected').width;
    const lh = fs + 10;
    ctx.fillStyle = '#ff3b30';
    ctx.fillRect(bb.x, Math.max(0, bb.y - lh), tw + 12, lh);
    ctx.fillStyle = '#fff';
    ctx.fillText('anomaly detected', bb.x + 6, Math.max(lh - 5, bb.y - 5));
    document.getElementById('results').classList.add('visible');
  };
  img.src = 'data:' + S.img2.mime + ';base64,' + S.img2.data;

  document.getElementById('jsonBody').innerHTML =
    `<span class="k">"bounding_box"</span>: {\n` +
    `  <span class="k">"x"</span>: <span class="n">${bb.x}</span>,\n` +
    `  <span class="k">"y"</span>: <span class="n">${bb.y}</span>,\n` +
    `  <span class="k">"width"</span>: <span class="n">${bb.width}</span>,\n` +
    `  <span class="k">"height"</span>: <span class="n">${bb.height}</span>\n}`;

  document.getElementById('descPanel').innerHTML =
    `<strong>Finding:</strong> ${data.description || 'See bounding box for detected anomaly.'}`;
}

function copyJ() {
  if (!S.lastJ) return;
  navigator.clipboard.writeText(JSON.stringify({ bounding_box: S.lastJ.bounding_box }, null, 2));
  const b = document.querySelector('.copy-btn');
  b.textContent = 'copied!'; setTimeout(() => b.textContent = 'copy', 1500);
}

function dlImg() {
  const a = document.createElement('a');
  a.href = document.getElementById('canvas').toDataURL('image/jpeg', 0.92);
  a.download = 'annotated_diff.jpg'; a.click();
}

function dlJSON() {
  if (!S.lastJ) return;
  const blob = new Blob([JSON.stringify({ bounding_box: S.lastJ.bounding_box }, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'bounding_box.json'; a.click();
}
</script>
</body>
</html>
"""


def run_web(port: int):
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # suppress default logging

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_PAGE.encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path != "/detect":
                self.send_response(404)
                self.end_headers()
                return

            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            api_key = body.get("api_key", "")
            prompt = body.get("prompt", DEFAULT_PROMPT)
            img1 = body.get("img1", {})
            img2 = body.get("img2", {})

            try:
                result = _call_claude_raw(
                    img1["data"], img1["mime"],
                    img2["data"], img2["mime"],
                    api_key, prompt,
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

    print(f"\n  Visual Diff Detector")
    print(f"  ─────────────────────────────────────")
    print(f"  Open in browser → http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")

    server = http.server.HTTPServer(("", port), Handler)
    server.serve_forever()


def _call_claude_raw(b64_1, mime_1, b64_2, mime_2, api_key, prompt):
    """Same as call_claude() but accepts pre-encoded base64 strings."""
    payload = {
        "model": "claude-opus-4-5",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime_1, "data": b64_1}},
                {"type": "image", "source": {"type": "base64", "media_type": mime_2, "data": b64_2}},
                {"type": "text", "text": prompt},
            ],
        }],
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
        body = e.read().decode()
        raise RuntimeError(f"API error {e.code}: {body}") from e

    text = "".join(b.get("text", "") for b in data.get("content", [])).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse Claude response:\n{text}") from e


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visual Diff Detector — powered by Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("image1", nargs="?", help="Path to first (before) image")
    parser.add_argument("image2", nargs="?", help="Path to second (after) image")
    parser.add_argument("--api-key", "-k", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--output", "-o", default="annotated_diff.jpg", help="Output image path (default: annotated_diff.jpg)")
    parser.add_argument("--web", action="store_true", help="Launch local web UI")
    parser.add_argument("--port", "-p", type=int, default=7860, help="Port for web UI (default: 7860)")
    args = parser.parse_args()

    if args.web:
        run_web(args.port)
    elif args.image1 and args.image2:
        run_cli(args)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  Web UI : python image_diff_detector.py --web")
        print("  CLI    : python image_diff_detector.py before.jpg after.jpg --api-key sk-ant-...")


if __name__ == "__main__":
    main()