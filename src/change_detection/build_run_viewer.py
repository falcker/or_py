#!/usr/bin/env python3
"""
Generate a self-contained HTML viewer for any run tree (sweep, demo, fp).

Walks <root>/**/run.json and renders one row per run with:
    references | target | annotated | prompt + result

Usage:
    python -m change_detection.build_run_viewer --root dev/demos_test
    python -m change_detection.build_run_viewer --root dev/sweeps --out dev/sweeps/runs.html

Images are linked by relative path (not embedded), so the HTML stays small
and you can rebuild quickly. Open the file directly in any browser; the
links resolve as long as the HTML lives at the root you scanned.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────
#  Discovery
# ─────────────────────────────────────────────

def find_runs(root: Path) -> list[dict]:
    """Find every (run_dir, manifest) pair under root. Tolerant of missing fields."""
    runs: list[dict] = []
    for rj in sorted(root.glob("**/run.json")):
        try:
            manifest = json.loads(rj.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] skipping {rj}: {e}", file=sys.stderr)
            continue
        run_dir = rj.parent

        copies = (manifest.get("inputs") or {}).get("copies") or {}
        ref_rels: list[str] = list(copies.get("refs") or [])
        cur_rel: str | None = copies.get("current")
        ann_rel: str | None = copies.get("annotated")

        def resolve(rel: str | None) -> Path | None:
            if not rel:
                return None
            try:
                p = (run_dir / rel).resolve()
                return p if p.exists() else None
            except Exception:
                return None

        refs = [p for p in (resolve(r) for r in ref_rels) if p]
        current = resolve(cur_rel)
        annotated = resolve(ann_rel)

        prompt_path = run_dir / "prompt.txt"
        prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

        anoms_path = run_dir / "anomalies.json"
        anomalies = []
        if anoms_path.exists():
            try:
                anomalies = json.loads(anoms_path.read_text(encoding="utf-8")) or []
            except Exception:
                pass

        target_dir = run_dir.parent
        scenario_dir = target_dir.parent

        runs.append({
            "run_dir": run_dir,
            "scenario": scenario_dir.name,
            "target_key": target_dir.name,
            "prompt": run_dir.name,
            "refs": refs,
            "current": current,
            "annotated": annotated,
            "prompt_text": prompt_text,
            "anomalies": anomalies,
            "usage": manifest.get("usage") or {},
            "manifest": manifest,
        })
    return runs


def rel_to(p: Path | None, base_dir: Path) -> str | None:
    """Return p as a forward-slash path relative to base_dir, or None."""
    if p is None:
        return None
    try:
        return p.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        # Different drive — fall back to absolute file:// URI
        return "file:///" + p.resolve().as_posix()


# ─────────────────────────────────────────────
#  Categorization (for badge colors)
# ─────────────────────────────────────────────

import re

OIL_RX = re.compile(
    r"(?<![a-z])(oil|hydrocarbon|petroleum|fuel|diesel|grease|crude|lubricant)(?![a-z])",
    re.IGNORECASE,
)
WATER_RX = re.compile(
    r"(?<![a-z])(water|wet|moisture|puddle|pool(?:ed|ing)?|fluid|liquid|spill|stain|leak|drip|damp|seep)(?![a-z])",
    re.IGNORECASE,
)


def classify(a: dict) -> str:
    """Return 'oil' | 'water' | 'other' from anomaly fields."""
    text = " ".join(filter(None, [
        a.get("anomaly_type"), a.get("description"), a.get("uncertainty"),
    ]))
    if OIL_RX.search(text):
        return "oil"
    if WATER_RX.search(text):
        return "water"
    return "other"


# ─────────────────────────────────────────────
#  Render
# ─────────────────────────────────────────────

PAGE_CSS = """
:root {
  --bg: #f3f4f7; --card: #fff; --border: #e3e5e8; --text: #1f2a44;
  --muted: #5e6878; --accent: #1f6feb;
  --oil: #c62828; --water: #1565c0; --other: #6b7280;
  --tp: #2e7d32; --fp: #c62828; --na: #6b7280;
  --sev-low: #777; --sev-medium: #b58900; --sev-high: #c62828;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       background: var(--bg); color: var(--text); }
header.top {
  position: sticky; top: 0; z-index: 10; background: var(--card);
  border-bottom: 1px solid var(--border); padding: 12px 20px;
  display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
}
header.top h1 { margin: 0; font-size: 16px; }
.filters { display: flex; gap: 8px; flex: 1; flex-wrap: wrap; }
.filters input, .filters select {
  padding: 6px 10px; border-radius: 5px; border: 1px solid var(--border);
  font-size: 13px; min-width: 0;
}
.filters input { flex: 1; min-width: 180px; }
.prompt-pickers {
  display: flex; gap: 4px; align-items: center; flex-wrap: wrap;
  padding: 4px 8px; border: 1px solid var(--border); border-radius: 5px;
}
.prompt-pickers .lbl {
  font-size: 11px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.04em; margin-right: 4px;
}
.prompt-pickers label {
  font-size: 12px; padding: 3px 8px; border-radius: 4px;
  background: #f3f4f7; cursor: pointer; user-select: none;
  border: 1px solid transparent;
}
.prompt-pickers label.on {
  background: #eef7e8; color: var(--tp); border-color: #b6dab0; font-weight: 600;
}
.prompt-pickers label input { display: none; }
.prompt-pickers .quick {
  font-size: 11px; color: var(--accent); background: transparent;
  border: 0; cursor: pointer; padding: 3px 6px;
}
.prompt-pickers .quick:hover { text-decoration: underline; }
.group-header {
  display: flex; gap: 8px; align-items: center; padding: 8px 12px;
  background: linear-gradient(90deg, #eef2fb, transparent);
  border-left: 3px solid var(--accent); border-radius: 4px;
  font-size: 13px; font-weight: 600; color: var(--text);
  margin-top: 6px;
}
.group-header .group-stats {
  margin-left: auto; font-weight: 400; color: var(--muted); font-size: 12px;
}
.row-label {
  display: inline-flex; gap: 2px; margin-left: auto;
  align-items: center;
}
.row-label .lbl {
  font-size: 10px; color: var(--muted); margin-right: 4px;
  text-transform: uppercase; letter-spacing: 0.04em;
}
.row-label button {
  background: transparent; border: 1px solid var(--border);
  color: var(--muted); padding: 2px 8px; border-radius: 4px;
  cursor: pointer; font-size: 11px; font-weight: 500;
  transition: all 0.1s ease;
}
.row-label button:hover { background: #f3f4f7; color: var(--text); }
.row-label button.on-TP {
  background: #e8f3e8; color: var(--tp); border-color: #b6dab0; font-weight: 600;
}
.row-label button.on-FP {
  background: #fbeaea; color: var(--fp); border-color: #d6a0a0; font-weight: 600;
}
.row-label button.on-Unclear {
  background: #faf3df; color: #8a6b00; border-color: #d6a019; font-weight: 600;
}
.run.is-labeled-TP { border-left: 3px solid var(--tp); }
.run.is-labeled-FP { border-left: 3px solid var(--fp); }
.run.is-labeled-Unclear { border-left: 3px solid #d6a019; }
.label-counts {
  display: inline-flex; gap: 10px; padding-left: 12px;
  border-left: 1px solid var(--border); color: var(--muted); font-size: 12px;
}
.label-counts b.tp { color: var(--tp); }
.label-counts b.fp { color: var(--fp); }
.label-counts b.un { color: #b58900; }
.stats { color: var(--muted); font-size: 13px; }
main { padding: 14px 18px; display: grid; gap: 14px; }

.run {
  background: var(--card); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 14px; display: grid; gap: 10px;
}
.run > header {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  border-bottom: 1px solid var(--border); padding-bottom: 8px;
}
.run > header .id {
  font-family: ui-monospace, monospace; font-size: 12px;
  color: var(--muted); flex: 1; word-break: break-all;
}
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 600; background: #e8eef7; color: var(--accent);
}
.badge.scenario { background: #e8eef7; color: var(--accent); }
.badge.target   { background: #f0e8f7; color: #6f42c1; }
.badge.prompt   { background: #eef7e8; color: var(--tp); }
.badge.zero     { background: #f3f4f7; color: var(--muted); }
.badge.has      { background: #fff4d6; color: #8a6b00; }
.badge.oil      { background: #ffe3e3; color: var(--oil); }
.badge.water    { background: #e3eeff; color: var(--water); }
.badge.other    { background: #eceff4; color: var(--other); }
.badge.sev-low    { background: #eceff4; color: var(--sev-low); }
.badge.sev-medium { background: #fff4d6; color: var(--sev-medium); }
.badge.sev-high   { background: #ffd6d6; color: var(--sev-high); }

.layout {
  display: grid; grid-template-columns: minmax(160px, 0.9fr) minmax(160px, 1fr) minmax(160px, 1fr) minmax(260px, 1.4fr);
  gap: 10px;
}
@media (max-width: 1100px) { .layout { grid-template-columns: 1fr 1fr; } }
@media (max-width: 700px)  { .layout { grid-template-columns: 1fr; } }

figure { margin: 0; display: flex; flex-direction: column; gap: 4px; }
figcaption {
  font-size: 11px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.04em;
}
.thumbs { display: flex; flex-direction: column; gap: 4px; }
.thumbs.row { flex-direction: row; flex-wrap: wrap; }
.thumbs img, .single img {
  width: 100%; max-height: 220px; object-fit: contain;
  background: #f8f9fb; border: 1px solid var(--border); border-radius: 6px;
  cursor: pointer;
}
.thumbs.row img { width: calc(50% - 2px); }
.placeholder {
  display: flex; align-items: center; justify-content: center;
  height: 100px; background: #f8f9fb; border: 1px dashed var(--border);
  border-radius: 6px; color: var(--muted); font-size: 12px; font-style: italic;
}
.result { display: flex; flex-direction: column; gap: 8px; }
.anomaly {
  border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px;
  font-size: 12.5px; background: #fcfcfd;
}
.anomaly .head { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; }
.anomaly .desc { line-height: 1.4; }
.anomaly .unc {
  margin-top: 6px; padding: 6px 8px; border-left: 3px solid #d6a019;
  background: #fff8e6; border-radius: 4px; font-size: 12px; line-height: 1.4;
}
.empty {
  padding: 10px; text-align: center; color: var(--muted); font-style: italic;
  font-size: 12.5px;
}
details { font-size: 12px; }
details summary {
  cursor: pointer; color: var(--accent); padding: 4px 0;
  font-weight: 600;
}
details pre {
  margin: 6px 0 0; padding: 10px; background: #fafbfd;
  border: 1px solid var(--border); border-radius: 6px;
  white-space: pre-wrap; word-break: break-word; max-height: 320px;
  overflow: auto; font-size: 11.5px; line-height: 1.45;
}
.hidden { display: none !important; }
"""

PAGE_JS = """
const $  = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

const LABEL_KEY = "runViewerLabels_" + (window.LABEL_NAMESPACE || "default");
function loadLabels() { try { return JSON.parse(localStorage.getItem(LABEL_KEY) || "{}"); } catch (e) { return {}; } }
function saveLabels(o) { localStorage.setItem(LABEL_KEY, JSON.stringify(o)); }

function selectedPrompts() {
  return new Set($$("#prompt-pickers input:checked").map(c => c.value));
}

function applyFilters() {
  const q   = $("#search").value.trim().toLowerCase();
  const sc  = $("#filter-scenario").value;
  const oa  = $("#only-anomalies").checked;
  const ps  = selectedPrompts();
  // toggle 'on' class on each checkbox label for styling
  $$("#prompt-pickers label").forEach(lbl => {
    const cb = lbl.querySelector("input");
    lbl.classList.toggle("on", cb && cb.checked);
  });
  let visible = 0;
  $$(".run").forEach(card => {
    const sc_ = card.dataset.scenario;
    const pr_ = card.dataset.prompt;
    const ac  = parseInt(card.dataset.anomalies, 10);
    const text = card.dataset.search;
    let show = true;
    if (sc && sc_ !== sc)       show = false;
    if (ps.size && !ps.has(pr_)) show = false;
    if (oa && ac === 0)         show = false;
    if (q && !text.includes(q)) show = false;
    card.classList.toggle("hidden", !show);
    if (show) visible++;
  });
  // hide group headers whose group has zero visible runs
  $$(".group-header").forEach(h => {
    const gid = h.dataset.group;
    const anyVisible = $$(`.run[data-group="${gid}"]`).some(
      r => !r.classList.contains("hidden")
    );
    h.classList.toggle("hidden", !anyVisible);
  });
  $("#visible-count").textContent = visible;
}

function setAllPrompts(state) {
  $$("#prompt-pickers input").forEach(c => { c.checked = state; });
  applyFilters();
}

function rowId(card) { return card.dataset.rowId; }

function refreshRowVisual(card) {
  const labels = loadLabels();
  const v = (labels[rowId(card)] || {}).label;
  card.classList.remove("is-labeled-TP", "is-labeled-FP", "is-labeled-Unclear");
  if (v) card.classList.add("is-labeled-" + v);
  card.querySelectorAll(".row-label button").forEach(b => {
    b.classList.toggle("on-" + b.dataset.label, b.dataset.label === v);
  });
}

function updateLabelCounts() {
  const labels = loadLabels();
  let tp = 0, fp = 0, un = 0, total = $$(".run").length;
  for (const k in labels) {
    if (labels[k].label === "TP") tp++;
    else if (labels[k].label === "FP") fp++;
    else if (labels[k].label === "Unclear") un++;
  }
  const el = $("#lbl-counts");
  if (el) {
    el.innerHTML = `<b class="tp">${tp}</b> TP · <b class="fp">${fp}</b> FP · <b class="un">${un}</b> ?  ·  ${total - tp - fp - un} unlabeled`;
  }
}

function setLabel(card, label) {
  const id = rowId(card);
  const labels = loadLabels();
  // Click again on the current choice = clear it
  if (labels[id] && labels[id].label === label) {
    delete labels[id];
  } else {
    labels[id] = { label, ts: Date.now() };
  }
  saveLabels(labels);
  refreshRowVisual(card);
  updateLabelCounts();
}

function exportLabels() {
  const labels = loadLabels();
  // Build an array with the prompt parsed out so aggregation by prompt is trivial.
  const rows = Object.entries(labels).map(([k, v]) => {
    const parts = k.split("/");
    const scenario = parts[0] || "";
    const target   = parts.length >= 3 ? parts.slice(1, -1).join("/") : (parts[1] || "");
    const prompt   = parts[parts.length - 1] || "";
    return {
      key: k, scenario, target, prompt,
      label: v.label, ts: v.ts, note: v.note || null,
    };
  });
  rows.sort((a, b) => a.key.localeCompare(b.key));
  // Quick per-prompt rollup for convenience
  const by_prompt = {};
  for (const r of rows) {
    const s = by_prompt[r.prompt] || (by_prompt[r.prompt] = {TP:0, FP:0, Unclear:0});
    if (s[r.label] !== undefined) s[r.label]++;
  }
  const out = {
    namespace: window.LABEL_NAMESPACE || "default",
    exported_at: new Date().toISOString(),
    total_labeled: rows.length,
    by_prompt,
    rows,
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = (window.LABEL_NAMESPACE || "labels") + ".json";
  a.click();
  URL.revokeObjectURL(url);
}

function clearAllLabels() {
  if (!confirm("Clear ALL labels for this view? (Cannot be undone.)")) return;
  localStorage.removeItem(LABEL_KEY);
  $$(".run").forEach(refreshRowVisual);
  updateLabelCounts();
}

document.addEventListener("DOMContentLoaded", () => {
  $("#search").addEventListener("input",  applyFilters);
  $("#filter-scenario").addEventListener("change", applyFilters);
  $("#only-anomalies").addEventListener("change", applyFilters);
  $$("#prompt-pickers input").forEach(c => c.addEventListener("change", applyFilters));
  $$("#prompt-pickers .quick").forEach(b => {
    b.addEventListener("click", e => setAllPrompts(b.dataset.action === "all"));
  });
  // Wire up per-row label buttons (delegated)
  document.body.addEventListener("click", e => {
    const btn = e.target.closest(".row-label button");
    if (!btn) return;
    const card = btn.closest(".run");
    if (card) setLabel(card, btn.dataset.label);
  });
  const exp = $("#export-labels");
  if (exp) exp.addEventListener("click", exportLabels);
  const clr = $("#clear-labels");
  if (clr) clr.addEventListener("click", clearAllLabels);
  // Initial paint
  $$(".run").forEach(refreshRowVisual);
  updateLabelCounts();
  applyFilters();
});
"""


def fmt_pct(v) -> str:
    if isinstance(v, (int, float)):
        return f"{v*100:.0f}%"
    return "—"


def render_thumb_strip(rel_paths: list[str | None], html_dir: Path, layout: str = "col") -> str:
    """Return HTML for a thumbnail strip given relative paths (already computed)."""
    rel_paths = [r for r in rel_paths if r]
    if not rel_paths:
        return '<div class="placeholder">(none)</div>'
    cls = "thumbs" + (" row" if layout == "row" and len(rel_paths) > 1 else "")
    parts = [f'<div class="{cls}">']
    for rp in rel_paths:
        rp_html = html.escape(rp)
        parts.append(
            f'<a href="{rp_html}" target="_blank"><img src="{rp_html}" loading="lazy"></a>'
        )
    parts.append("</div>")
    return "".join(parts)


def render_run(run: dict, html_dir: Path) -> str:
    refs_rel = [rel_to(p, html_dir) for p in run["refs"]]
    cur_rel  = rel_to(run["current"], html_dir)
    ann_rel  = rel_to(run["annotated"], html_dir)

    # Anomaly count + class breakdown
    anoms = run["anomalies"]
    classes = [classify(a) for a in anoms]
    n_oil   = sum(1 for c in classes if c == "oil")
    n_water = sum(1 for c in classes if c == "water")
    n_other = sum(1 for c in classes if c == "other")

    # Header badges
    header_bits = [
        f'<span class="badge scenario">{html.escape(run["scenario"])}</span>',
        f'<span class="badge target">{html.escape(run["target_key"])}</span>',
        f'<span class="badge prompt">{html.escape(run["prompt"])}</span>',
    ]
    if not anoms:
        header_bits.append('<span class="badge zero">0 anomalies</span>')
    else:
        header_bits.append(f'<span class="badge has">{len(anoms)} anomaly/anomalies</span>')
        if n_oil:   header_bits.append(f'<span class="badge oil">{n_oil}× oil</span>')
        if n_water: header_bits.append(f'<span class="badge water">{n_water}× water</span>')
        if n_other: header_bits.append(f'<span class="badge other">{n_other}× other</span>')
    usage = run["usage"]
    if usage.get("total_tokens"):
        header_bits.append(f'<span class="id">{usage["total_tokens"]} tok · model={html.escape(str(usage.get("model","?")))}</span>')

    # References
    refs_block = (
        f'<figure><figcaption>References ({len(run["refs"])})</figcaption>'
        f'{render_thumb_strip(refs_rel, html_dir, "row")}</figure>'
    )
    target_block = (
        f'<figure><figcaption>Target</figcaption>'
        f'{render_thumb_strip([cur_rel], html_dir, "col")}</figure>'
    )
    if ann_rel:
        annotated_block = (
            f'<figure><figcaption>Annotated</figcaption>'
            f'{render_thumb_strip([ann_rel], html_dir, "col")}</figure>'
        )
    else:
        annotated_block = (
            f'<figure><figcaption>Annotated</figcaption>'
            f'<div class="placeholder">no detections</div></figure>'
        )

    # Result column: anomalies + prompt collapsible
    if anoms:
        cards = []
        for a in anoms:
            kind = classify(a)
            t = html.escape(a.get("anomaly_type") or "anomaly")
            sev = (a.get("severity") or "low").lower()
            conf = a.get("confidence")
            conf_s = fmt_pct(conf)
            badge_row = (
                f'<span class="badge {kind}">{t}</span>'
                f'<span class="badge sev-{html.escape(sev)}">{html.escape(sev)}</span>'
                f'<span class="badge">conf {html.escape(conf_s)}</span>'
            )
            desc = html.escape(a.get("description") or "(no description)")
            unc_html = ""
            if a.get("uncertainty"):
                unc_html = f'<div class="unc">{html.escape(a["uncertainty"])}</div>'
            cards.append(
                f'<div class="anomaly"><div class="head">{badge_row}</div>'
                f'<div class="desc">{desc}</div>{unc_html}</div>'
            )
        anoms_html = "".join(cards)
    else:
        anoms_html = '<div class="empty">No anomalies detected</div>'

    prompt_excerpt = run["prompt_text"]
    prompt_block = ""
    if prompt_excerpt:
        prompt_block = (
            f'<details><summary>Prompt ({len(prompt_excerpt)} chars)</summary>'
            f'<pre>{html.escape(prompt_excerpt)}</pre></details>'
        )

    result_block = (
        f'<div class="result">{anoms_html}{prompt_block}</div>'
    )

    # Data attributes for filtering
    search_text = " ".join([
        run["scenario"], run["target_key"], run["prompt"],
        *(a.get("anomaly_type", "") for a in anoms),
        *(a.get("description", "") for a in anoms),
    ]).lower()

    group_id = f"{run['scenario']}__{run['target_key']}"
    row_id = f"{run['scenario']}/{run['target_key']}/{run['prompt']}"
    label_pills = (
        '<div class="row-label" title="Mark this row (click again to clear)">'
        '<span class="lbl">judge</span>'
        '<button type="button" data-label="TP">TP</button>'
        '<button type="button" data-label="FP">FP</button>'
        '<button type="button" data-label="Unclear">?</button>'
        '</div>'
    )
    return (
        f'<article class="run" '
        f'data-scenario="{html.escape(run["scenario"])}" '
        f'data-prompt="{html.escape(run["prompt"])}" '
        f'data-group="{html.escape(group_id)}" '
        f'data-row-id="{html.escape(row_id, quote=True)}" '
        f'data-anomalies="{len(anoms)}" '
        f'data-search="{html.escape(search_text, quote=True)}">'
        f'<header>{"".join(header_bits)}{label_pills}</header>'
        f'<div class="layout">{refs_block}{target_block}{annotated_block}{result_block}</div>'
        f'</article>'
    )


def render_page(runs: list[dict], root: Path, html_dir: Path, title: str) -> str:
    # Sort so prompts for the same case are adjacent (and groups are sorted).
    runs = sorted(runs, key=lambda r: (r["scenario"], r["target_key"], r["prompt"]))

    scenarios = sorted({r["scenario"] for r in runs})
    prompts   = sorted({r["prompt"] for r in runs})

    opts_sc = "".join(f'<option value="{html.escape(s)}">{html.escape(s)}</option>' for s in scenarios)

    # One checkbox per prompt — all checked by default. Click to toggle.
    prompt_picker_html = (
        '<div class="prompt-pickers" id="prompt-pickers">'
        '<span class="lbl">Compare prompts:</span>'
        + "".join(
            f'<label class="on"><input type="checkbox" value="{html.escape(p)}" checked> '
            f'{html.escape(p)}</label>'
            for p in prompts
        )
        + '<button type="button" class="quick" data-action="all">all</button>'
        + '<button type="button" class="quick" data-action="none">none</button>'
        + '</div>'
    )

    # Build body: emit a group-header before the first run of each (scenario, target),
    # then the run cards themselves.
    body_parts: list[str] = []
    prev_group = None
    for r in runs:
        gid = f"{r['scenario']}__{r['target_key']}"
        if gid != prev_group:
            # Stats for the group
            group_runs = [x for x in runs
                          if x["scenario"] == r["scenario"] and x["target_key"] == r["target_key"]]
            n_runs = len(group_runs)
            n_with = sum(1 for x in group_runs if x["anomalies"])
            body_parts.append(
                f'<div class="group-header" data-group="{html.escape(gid)}">'
                f'<span>{html.escape(r["scenario"])} / {html.escape(r["target_key"])}</span>'
                f'<span class="group-stats">{n_runs} prompt(s) · {n_with} with detections</span>'
                f'</div>'
            )
            prev_group = gid
        body_parts.append(render_run(r, html_dir))

    body = "\n".join(body_parts)
    n_with = sum(1 for r in runs if r["anomalies"])
    tokens = sum((r["usage"].get("total_tokens") or 0) for r in runs)
    # Namespace for localStorage so different viewer files don't share labels
    label_namespace = root.resolve().as_posix().replace("/", "_").replace(":", "")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<header class="top">
  <h1>{html.escape(title)}</h1>
  <div class="filters">
    <input id="search" placeholder="Filter (scenario / target / prompt / type / description)…">
    <select id="filter-scenario">
      <option value="">All scenarios ({len(scenarios)})</option>{opts_sc}
    </select>
    {prompt_picker_html}
    <label><input type="checkbox" id="only-anomalies"> Only with anomalies</label>
  </div>
  <div class="stats">
    <b id="visible-count">{len(runs)}</b> of <b id="total-count">{len(runs)}</b> runs
    · {n_with} with detections
    · {tokens:,} tokens total
    <span class="label-counts" id="lbl-counts"></span>
    <button id="export-labels" class="quick" type="button" style="margin-left:8px">Export labels JSON</button>
    <button id="clear-labels"  class="quick" type="button">Clear</button>
  </div>
</header>
<script>window.LABEL_NAMESPACE = {label_namespace!r};</script>
<main>
{body}
</main>
<script>{PAGE_JS}</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate an HTML viewer for a run tree (sweep / demo / fp)."
    )
    ap.add_argument("--root", required=True,
                    help="Root containing the run tree (e.g. dev/demos_test, dev/sweeps).")
    ap.add_argument("--out", default=None,
                    help="Output HTML path. Default: <root>/runs_viewer.html")
    ap.add_argument("--title", default=None,
                    help="Page title. Default derived from root.")
    ap.add_argument("--prompts", default=None,
                    help="Optional comma-separated prompt-stem prefix filter.")
    ap.add_argument("--scenarios", default=None,
                    help="Optional comma-separated scenario-name filter.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"Error: root '{root}' is not a directory.")

    out_path = Path(args.out) if args.out else (root / "runs_viewer.html")
    html_dir = out_path.parent
    html_dir.mkdir(parents=True, exist_ok=True)

    runs = find_runs(root)
    if not runs:
        sys.exit(f"No run.json files found under {root}.")

    if args.prompts:
        wanted = {s.strip() for s in args.prompts.split(",") if s.strip()}
        runs = [r for r in runs
                if any(r["prompt"].startswith(w) for w in wanted)]
    if args.scenarios:
        wanted = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        runs = [r for r in runs if r["scenario"] in wanted]

    if not runs:
        sys.exit("No runs left after applying --prompts / --scenarios filters.")

    title = args.title or f"Run viewer — {root.as_posix()}"
    page = render_page(runs, root, html_dir, title)
    out_path.write_text(page, encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path}  ({size_kb:,.0f} KB, {len(runs)} runs)")
    print(f"Open it in your browser. Images load from disk relative to the HTML location.")


if __name__ == "__main__":
    main()
