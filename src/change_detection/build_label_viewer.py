#!/usr/bin/env python3
"""
Generate a self-contained HTML labeling viewer for a chosen prompt's
detections from an FP test or sweep tree.

Usage:
    python -m change_detection.build_label_viewer \\
        --root dev/fp_test \\
        --prompt 02_basic \\
        --out  dev/fp_test/label_02_basic.html

Open the resulting HTML in any browser. Each detected anomaly is shown with
its annotated image and metadata. For each one you pick TP / FP / Unclear and
optionally type a note. Decisions persist in localStorage between sessions.
Click "Download labels JSON" to save them to disk.

Sister script `apply_labels.py` (not built yet) can merge a labels JSON back
into the anomaly records or compute precision/recall summaries.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import sys
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
    --bg: #f7f8fa;
    --card: #fff;
    --border: #e3e5e8;
    --text: #1f2a44;
    --muted: #5a6478;
    --tp: #2e7d32;
    --fp: #c62828;
    --unclear: #c69214;
    --accent: #1f6feb;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0;
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text);
  }}
  header {{
    position: sticky; top: 0; z-index: 5;
    background: var(--card); border-bottom: 1px solid var(--border);
    padding: 14px 24px; display: flex; gap: 24px;
    align-items: center; justify-content: space-between;
    flex-wrap: wrap;
  }}
  header h1 {{ margin: 0; font-size: 18px; }}
  header .summary {{ color: var(--muted); font-size: 14px; }}
  header button {{
    background: var(--accent); color: #fff; border: 0; border-radius: 6px;
    padding: 8px 14px; font-size: 13px; cursor: pointer;
  }}
  header button.ghost {{ background: transparent; color: var(--accent); border: 1px solid var(--accent); }}
  main {{ padding: 18px 24px; display: grid; gap: 18px; }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
    display: grid; grid-template-columns: minmax(420px, 1fr) 380px; gap: 18px;
  }}
  @media (max-width: 1000px) {{ .card {{ grid-template-columns: 1fr; }} }}
  .card .img-wrap {{ position: relative; }}
  .card img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
  .card h2 {{ margin: 0 0 6px; font-size: 14px; font-weight: 600; }}
  .card .asset {{ font-family: ui-monospace, monospace; font-size: 11px; color: var(--muted); word-break: break-all; }}
  .meta {{ display: grid; gap: 6px; font-size: 13px; margin-top: 8px; }}
  .meta .k {{ color: var(--muted); margin-right: 6px; }}
  .meta .desc {{ background: #f1f4f8; padding: 8px 10px; border-radius: 6px; font-size: 12.5px; line-height: 1.4; }}
  .meta .unc  {{ background: #fff8e6; padding: 8px 10px; border-radius: 6px;
                 font-size: 12.5px; line-height: 1.4; border-left: 3px solid #d6a019; }}
  .labels {{ margin-top: 12px; display: flex; gap: 6px; flex-wrap: wrap; }}
  .labels label {{
    cursor: pointer; padding: 6px 12px; border-radius: 6px;
    border: 1px solid var(--border); font-size: 13px; user-select: none;
  }}
  .labels input {{ display: none; }}
  .labels input:checked + span {{ color: #fff; font-weight: 600; }}
  .labels .tp input:checked + span      {{ background: var(--tp); color: #fff; }}
  .labels .fp input:checked + span      {{ background: var(--fp); color: #fff; }}
  .labels .unclear input:checked + span {{ background: var(--unclear); color: #fff; }}
  .labels label span {{ display: inline-block; padding: 6px 10px; margin: -6px -10px; border-radius: 4px; }}
  textarea {{ width: 100%; margin-top: 8px; min-height: 60px; padding: 8px;
              border-radius: 6px; border: 1px solid var(--border);
              font-family: inherit; font-size: 13px; resize: vertical; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px;
            font-size: 11px; font-weight: 600; margin-right: 4px;
            background: #e8eef7; color: var(--accent); }}
  .badge.oil    {{ background: #ffe6e6; color: var(--fp); }}
  .badge.water  {{ background: #e6f0ff; color: var(--accent); }}
  .badge.equip  {{ background: #ecf3ec; color: var(--tp); }}
  .badge.sev-low    {{ background: #e8eef7; color: #555; }}
  .badge.sev-medium {{ background: #fff4d6; color: #8a6b00; }}
  .badge.sev-high   {{ background: #ffd6d6; color: var(--fp); }}
  .counts {{ display: flex; gap: 16px; font-size: 13px; }}
  .counts b.tp {{ color: var(--tp); }}
  .counts b.fp {{ color: var(--fp); }}
  .counts b.unclear {{ color: var(--unclear); }}
</style>
</head>
<body>
<header>
  <div>
    <h1>{title}</h1>
    <div class="summary">{summary_line}</div>
  </div>
  <div class="counts">
    <span>TP: <b class="tp" id="cTP">0</b></span>
    <span>FP: <b class="fp" id="cFP">0</b></span>
    <span>Unclear: <b class="unclear" id="cUN">0</b></span>
    <span>Unlabeled: <b id="cNA">{n_items}</b></span>
  </div>
  <div>
    <button class="ghost" onclick="resetLabels()">Reset</button>
    <button onclick="exportLabels()">Download labels JSON</button>
  </div>
</header>

<main id="grid">
{cards}
</main>

<script>
  const STORAGE_KEY = "{storage_key}";
  const N_ITEMS = {n_items};

  function load() {{
    try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}") }}
    catch (e) {{ return {{}} }}
  }}
  function save(state) {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }}

  function restore() {{
    const state = load();
    for (const id in state) {{
      const v = state[id];
      const r = document.querySelector(`input[name="lbl-${{id}}"][value="${{v.label}}"]`);
      if (r) r.checked = true;
      const t = document.getElementById(`note-${{id}}`);
      if (t && v.note) t.value = v.note;
    }}
    updateCounts();
  }}

  function updateCounts() {{
    const state = load();
    let tp = 0, fp = 0, un = 0;
    for (const id in state) {{
      if (state[id].label === "TP") tp++;
      else if (state[id].label === "FP") fp++;
      else if (state[id].label === "Unclear") un++;
    }}
    document.getElementById("cTP").textContent = tp;
    document.getElementById("cFP").textContent = fp;
    document.getElementById("cUN").textContent = un;
    document.getElementById("cNA").textContent = N_ITEMS - (tp + fp + un);
  }}

  function onLabel(id, label) {{
    const state = load();
    state[id] = {{ ...(state[id] || {{}}), label }};
    save(state); updateCounts();
  }}
  function onNote(id, note) {{
    const state = load();
    state[id] = {{ ...(state[id] || {{}}), note }};
    save(state);
  }}
  function resetLabels() {{
    if (!confirm("Clear all labels for this view?")) return;
    localStorage.removeItem(STORAGE_KEY);
    document.querySelectorAll('input[type="radio"]').forEach(r => r.checked = false);
    document.querySelectorAll('textarea').forEach(t => t.value = "");
    updateCounts();
  }}
  function exportLabels() {{
    const state = load();
    const blob = new Blob([JSON.stringify(state, null, 2)], {{type: "application/json"}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "{storage_key}.json";
    a.click();
    URL.revokeObjectURL(url);
  }}

  document.addEventListener("DOMContentLoaded", restore);
</script>
</body>
</html>
"""


CARD_TEMPLATE = """
<div class="card" id="card-{id}">
  <div class="img-wrap"><img src="data:{mime};base64,{b64}" alt="Annotated detection"></div>
  <div>
    <h2>{type_html}<span class="badge sev-{severity_class}">{severity_html}</span><span class="badge">conf {confidence}</span></h2>
    <div class="asset">{asset_html}</div>
    <div class="meta">
      <div><span class="k">Description:</span></div>
      <div class="desc">{description_html}</div>
      {uncertainty_block}
      <div><span class="k">Run dir:</span> <code>{rundir_html}</code></div>
    </div>
    <div class="labels">
      <label class="tp"><input type="radio" name="lbl-{id}" value="TP" onclick="onLabel('{id}','TP')"><span>True positive</span></label>
      <label class="fp"><input type="radio" name="lbl-{id}" value="FP" onclick="onLabel('{id}','FP')"><span>False positive</span></label>
      <label class="unclear"><input type="radio" name="lbl-{id}" value="Unclear" onclick="onLabel('{id}','Unclear')"><span>Unclear</span></label>
    </div>
    <textarea id="note-{id}" placeholder="Optional note..." oninput="onNote('{id}', this.value)"></textarea>
  </div>
</div>
"""


def b64_image(path: Path) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/jpeg"
    return mime, base64.b64encode(path.read_bytes()).decode("ascii")


def kind_badge(anomaly_type: str) -> str:
    t = (anomaly_type or "").lower()
    if "oil" in t or "hydrocarbon" in t or "fuel" in t:
        return "oil"
    if "water" in t or "wet" in t or "fluid" in t or "spill" in t or "leak" in t or "stain" in t:
        return "water"
    return "equip"


def severity_class(s: str) -> str:
    s = (s or "").lower()
    if s in ("low", "medium", "high"):
        return s
    return "low"


def discover_detections(root: Path, prompt: str) -> list[dict]:
    """Walk the FP-test / sweep tree and return one entry per anomaly for the
    given prompt stem."""
    items: list[dict] = []
    for run_dir in sorted(root.glob(f"*/{prompt}")):
        a_path = run_dir / "anomalies.json"
        if not a_path.exists():
            continue
        anomalies = json.loads(a_path.read_text(encoding="utf-8"))
        if not anomalies:
            continue
        annotated = run_dir / "annotated.jpg"
        if not annotated.exists():
            # No annotated image — skip (means no anomalies were drawn)
            continue
        asset_name = run_dir.parent.name
        for i, a in enumerate(anomalies, start=1):
            items.append({
                "id": f"{asset_name}__{prompt}__{i}",
                "asset": asset_name,
                "prompt": prompt,
                "image_path": annotated,
                "run_dir": run_dir,
                "anomaly_index": i,
                "anomaly": a,
            })
    return items


def render_html(items: list[dict], title: str, summary_line: str,
                storage_key: str) -> str:
    cards = []
    for it in items:
        a = it["anomaly"]
        mime, b64 = b64_image(it["image_path"])
        type_str = a.get("anomaly_type") or "anomaly"
        kind = kind_badge(type_str)
        sev = a.get("severity") or "low"
        conf = a.get("confidence")
        conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else "—"
        type_html = f'<span class="badge {kind}">{html.escape(type_str)}</span>'
        unc = a.get("uncertainty")
        unc_block = (
            f'<div><span class="k">Uncertainty:</span></div>'
            f'<div class="unc">{html.escape(unc)}</div>'
        ) if unc else ""
        cards.append(CARD_TEMPLATE.format(
            id=html.escape(it["id"]),
            mime=mime,
            b64=b64,
            type_html=type_html,
            severity_html=html.escape(sev),
            severity_class=severity_class(sev),
            confidence=html.escape(conf_str),
            asset_html=html.escape(it["asset"]),
            description_html=html.escape(a.get("description") or ""),
            uncertainty_block=unc_block,
            rundir_html=html.escape(str(it["run_dir"])),
        ))
    return HTML_TEMPLATE.format(
        title=html.escape(title),
        summary_line=html.escape(summary_line),
        cards="\n".join(cards),
        n_items=len(items),
        storage_key=storage_key,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a self-contained HTML labeling viewer.",
    )
    ap.add_argument("--root", default="dev/fp_test",
                    help="Tree to scan for runs (default: dev/fp_test).")
    ap.add_argument("--prompt", default="02_basic",
                    help="Prompt stem to filter detections by (default: 02_basic).")
    ap.add_argument("--out", default=None,
                    help="Output HTML path. Default: <root>/label_<prompt>.html")
    ap.add_argument("--storage-key", default=None,
                    help="LocalStorage key for label persistence. Default derived from prompt + root.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"Error: root '{root}' is not a directory.")

    items = discover_detections(root, args.prompt)
    if not items:
        sys.exit(f"No detections found for prompt '{args.prompt}' under {root}.")

    out_path = Path(args.out) if args.out else (root / f"label_{args.prompt}.html")
    storage_key = args.storage_key or f"labels_{root.name}_{args.prompt}"

    n_assets = len({it["asset"] for it in items})
    title = f"Label viewer — {args.prompt} ({root.name})"
    summary_line = (f"{len(items)} anomaly/anomalies across {n_assets} asset(s). "
                    f"Labels persist in your browser; click Download to export.")
    html_str = render_html(items, title, summary_line, storage_key)
    out_path.write_text(html_str, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path}  ({size_mb:.1f} MB, {len(items)} cards across {n_assets} assets)")
    print(f"Storage key: {storage_key}")
    print("Open it in any browser. Decisions auto-save; click 'Download labels JSON' when done.")


if __name__ == "__main__":
    main()
