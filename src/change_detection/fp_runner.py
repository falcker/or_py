#!/usr/bin/env python3
"""
False-positive test for oil-detection prompts on a "known-no-oil" image set.

Folder layout (e.g. dev/Original_1000x750/):
    <root>/
        <asset_folder_name>/         # ONE physical asset; unique by full
            img_01.jpg               # folder name (NOT by code suffix).
            img_02.jpg               # ~12 images per asset, same object.
            ...

For each sampled asset we:
    - sort images by name (oldest -> newest)
    - use the first 2 images as references
    - use the last image as the target
    - run each selected prompt
    - flag any anomaly whose type/description mentions oil/hydrocarbon as a
      false positive (the dataset is known to contain no oil spillage anywhere)

CRITICAL: refs and target always come from the SAME subdirectory. The suffix
codes (MH/V/F/RC/RM/...) are TYPE tags, not identity — only the full folder
name is unique.

Usage:
    python -m change_detection.fp_runner \\
        --root dev/Original_1000x750 \\
        --n-assets 30 \\
        --prompts 00_unconditional,02_basic,05_rigorous

Produces:
    dev/fp_test/
        fp_summary.json                 — all run rows + aggregate counts
        <asset>/inputs/                 — shared input copies
        <asset>/<prompt>/               — annotated.jpg, anomalies.json, run.json, ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from change_detection.claude_change_detect import (
    annotate_image,
    assemble_prompt,
    call_claude,
    copy_inputs_to,
    create_run_folder,
    load_prompt,
)

load_dotenv()


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# Anomaly types or description keywords that count as an oil-related FP.
# Uses letter-class lookarounds (not \b) because anomaly_type values like
# "oil_stain" have '_' which is a \w char — \b would fail to match.
OIL_RX = re.compile(
    r"(?<![a-z])(oil|hydrocarbon|petroleum|fuel|diesel|grease|crude|lubricant)(?![a-z])",
    re.IGNORECASE,
)
ACTIVE_LEAK_TYPES = {"active_leak", "leak", "spill", "drip"}


# ─────────────────────────────────────────────
#  Asset discovery
# ─────────────────────────────────────────────

def list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


_BAD_CHARS = set("](")  # extraction-artifact folder names like "34_544_RM2](1_..."


def discover_assets(root: Path, min_images: int = 3) -> list[dict]:
    """Return one entry per usable asset subdirectory.

    Skips empty dirs, dirs with too few images, and folders whose names look
    like Windows extraction duplicates (contain ']' or '(').
    """
    assets: list[dict] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if any(c in d.name for c in _BAD_CHARS):
            continue
        imgs = list_images(d)
        if len(imgs) < min_images:
            continue
        assets.append({
            "name": d.name,
            "path": d,
            "images": imgs,
            # Pull the short type tag (e.g. "MH", "V", "F") for stratified sampling
            # / per-type aggregation; NOT used for identity.
            "type_tag": _extract_type_tag(d.name),
        })
    return assets


def _extract_type_tag(name: str) -> str:
    """Best-effort short type tag extracted from a folder name like
    '12_542_V_5b2c3c19...' -> 'V'. Falls back to '?' if not parseable."""
    parts = name.split("_")
    # Pattern is roughly <seq>_<unit>_<TAG>_<hash>; tag is the part right
    # before a 16+ hex-ish chunk.
    for i, p in enumerate(parts):
        if re.fullmatch(r"[0-9a-f]{16,}", p, re.IGNORECASE) and i > 0:
            return parts[i - 1].split("-")[0]
    # Fallback: 3rd segment if present
    return parts[2] if len(parts) >= 3 else "?"


# ─────────────────────────────────────────────
#  FP scoring
# ─────────────────────────────────────────────

def is_oil_fp(anomaly: dict) -> bool:
    """True if an anomaly is plausibly claiming oil/hydrocarbon presence."""
    t = (anomaly.get("anomaly_type") or "").lower()
    if OIL_RX.search(t):
        return True
    if t in ACTIVE_LEAK_TYPES:
        # Generic 'leak' / 'spill' is ambiguous; only count if description also
        # mentions oil/hydrocarbon to avoid penalising water-leak detections.
        desc = (anomaly.get("description") or "").lower()
        if OIL_RX.search(desc):
            return True
    desc = (anomaly.get("description") or "").lower()
    if OIL_RX.search(desc):
        return True
    return False


# ─────────────────────────────────────────────
#  Prompt resolution
# ─────────────────────────────────────────────

def resolve_prompts(prompts_dir: Path, names: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in names:
        stem = raw.strip()
        if not stem:
            continue
        cands = sorted(prompts_dir.glob(f"{stem}*.txt"))
        if not cands:
            sys.exit(f"Prompt '{stem}' not found in {prompts_dir}.")
        out.append(cands[0])
    return out


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Oil-detection false-positive test on a known-no-oil dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--root", default="dev/Original_1000x750",
                    help="Dataset root containing one subdirectory per asset.")
    ap.add_argument("--out", default="dev/fp_test",
                    help="Output root for per-run artifacts and the summary JSON.")
    ap.add_argument("--prompts-dir", default="src/change_detection/prompts",
                    help="Directory of prompt .txt files.")
    ap.add_argument("--prompts", default="03,04,05",
                    help="Comma-separated prompt stems (prefix match allowed). "
                         "Default: 03,04,05 → 03_focused, 04_calibrated, 05_rigorous.")
    ap.add_argument("--n-assets", type=int, default=30,
                    help="How many assets (subdirs) to sample. Use 0 for all.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for sampling. Stable across runs.")
    ap.add_argument("--n-refs", type=int, default=2,
                    help="How many of the asset's images to use as references.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the sampling plan; do not call the API.")
    ap.add_argument("--api-key", "-k", default=None,
                    help="Anthropic API key (overrides ANTHROPIC_API_KEY).")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"Error: root '{root}' is not a directory.")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    prompts_dir = Path(args.prompts_dir)
    if not prompts_dir.is_dir():
        sys.exit(f"Error: prompts-dir '{prompts_dir}' is not a directory.")

    prompt_paths = resolve_prompts(prompts_dir,
                                   [p for p in args.prompts.split(",") if p.strip()])

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("Error: provide --api-key or set ANTHROPIC_API_KEY.")

    rng = random.Random(args.seed)
    assets = discover_assets(root)
    if not assets:
        sys.exit(f"No eligible assets in {root}.")
    if args.n_assets <= 0 or args.n_assets >= len(assets):
        sampled = assets
    else:
        sampled = rng.sample(assets, args.n_assets)
    sampled.sort(key=lambda a: a["name"])

    total = len(sampled) * len(prompt_paths)
    print(f"FP test plan:")
    print(f"  root      : {root}")
    print(f"  eligible  : {len(assets)} assets")
    print(f"  sampled   : {len(sampled)} assets (seed={args.seed})")
    print(f"  prompts   : {[p.stem for p in prompt_paths]}")
    print(f"  per asset : {args.n_refs} ref(s) + 1 target, all from same subdir")
    print(f"  calls     : {total}")
    print(f"  output    : {out_root}")

    type_counts = defaultdict(int)
    for a in sampled:
        type_counts[a["type_tag"]] += 1
    print(f"  by tag    : {dict(sorted(type_counts.items()))}")
    print()

    if args.dry_run:
        for a in sampled[:20]:
            print(f"  - {a['name']}  ({len(a['images'])} imgs, tag={a['type_tag']})")
        if len(sampled) > 20:
            print(f"  ... and {len(sampled) - 20} more")
        print("\nDry run — exiting before any API calls.")
        return

    rows: list[dict] = []
    done = 0
    for asset in sampled:
        imgs = asset["images"]
        n_refs = max(1, min(args.n_refs, len(imgs) - 1))
        refs = [str(p) for p in imgs[:n_refs]]
        target = str(imgs[-1])

        asset_dir = out_root / asset["name"]
        asset_dir.mkdir(parents=True, exist_ok=True)
        # Shared inputs/ folder for all sibling prompts under this asset.
        copy_inputs_to(asset_dir, refs, [], target)

        for pp in prompt_paths:
            done += 1
            label = f"{asset['name']}__{pp.stem}"
            print(f"[{done}/{total}] {label} ...", end=" ", flush=True)
            try:
                base = load_prompt(str(pp))
                final = assemble_prompt(base, refs, [], target)
                result, usage = call_claude(refs + [target], api_key, final)
                anomalies = result.get("anomalies", [])
                oil_fp = [a for a in anomalies if is_oil_fp(a)]

                run_dir = asset_dir / pp.stem
                run_dir.mkdir(parents=True, exist_ok=True)
                annotated = run_dir / "annotated.jpg"
                out_img = annotate_image(target, anomalies, str(annotated)) if anomalies else ""

                create_run_folder(
                    runs_dir=str(asset_dir),
                    label=label,
                    ref_paths=refs,
                    examples=[],
                    current_path=target,
                    prompt=final,
                    anomalies=anomalies,
                    usage=usage,
                    annotated_image_path=str(annotated) if out_img else None,
                    copy_inputs=False,
                    folder_name=pp.stem,
                    external_inputs_dir=asset_dir / "inputs",
                )

                rows.append({
                    "asset": asset["name"],
                    "type_tag": asset["type_tag"],
                    "prompt": pp.stem,
                    "anomaly_count": len(anomalies),
                    "oil_fp_count": len(oil_fp),
                    "oil_fp_types": [a.get("anomaly_type") for a in oil_fp],
                    "total_tokens": usage.get("total_tokens"),
                    "run_dir": str(run_dir),
                })
                tag = "" if not oil_fp else f"  ⚑ {len(oil_fp)} OIL-FP"
                print(f"OK ({len(anomalies)} total, "
                      f"{usage.get('total_tokens')} tok){tag}")
            except Exception as e:
                print(f"FAIL ({type(e).__name__}: {e})")
                traceback.print_exc()
                rows.append({
                    "asset": asset["name"],
                    "type_tag": asset["type_tag"],
                    "prompt": pp.stem,
                    "error": f"{type(e).__name__}: {e}",
                })

    # Aggregate
    by_prompt: dict[str, dict] = defaultdict(
        lambda: {"ran": 0, "with_anomaly": 0, "with_oil_fp": 0, "tokens": 0,
                 "oil_fp_anomalies": 0}
    )
    for r in rows:
        if "error" in r:
            continue
        s = by_prompt[r["prompt"]]
        s["ran"] += 1
        if r["anomaly_count"] > 0:
            s["with_anomaly"] += 1
        if r["oil_fp_count"] > 0:
            s["with_oil_fp"] += 1
        s["oil_fp_anomalies"] += r["oil_fp_count"]
        s["tokens"] += r.get("total_tokens") or 0

    summary_obj = {
        "config": {
            "root": str(root),
            "seed": args.seed,
            "n_refs": args.n_refs,
            "sampled": len(sampled),
            "eligible": len(assets),
            "prompts": [p.stem for p in prompt_paths],
        },
        "by_prompt": dict(by_prompt),
        "rows": rows,
    }
    (out_root / "fp_summary.json").write_text(
        json.dumps(summary_obj, indent=2), encoding="utf-8"
    )

    print()
    print("===== FP test complete =====")
    header = f"{'Prompt':<22} {'Ran':>4} {'AnyAnom':>8} {'AnyAnom%':>9} {'OilFP':>6} {'OilFP%':>7} {'Tokens':>9}"
    print(header)
    print("-" * len(header))
    for prompt, s in by_prompt.items():
        ran = s["ran"] or 1
        any_pct = 100 * s["with_anomaly"] / ran
        fp_pct = 100 * s["with_oil_fp"] / ran
        print(f"{prompt:<22} {s['ran']:>4} {s['with_anomaly']:>8} "
              f"{any_pct:>8.1f}% {s['with_oil_fp']:>6} {fp_pct:>6.1f}% {s['tokens']:>9,}")
    print()
    print(f"Summary written to {out_root / 'fp_summary.json'}")


if __name__ == "__main__":
    main()
