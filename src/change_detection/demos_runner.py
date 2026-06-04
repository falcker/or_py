#!/usr/bin/env python3
"""
Run prompts against the labeled demo cases and compute real precision/recall
against the per-case ground truth encoded in the folder name.

Expected layout:
    <root>/<category>/<case_dir>/
        <ref_image>      sorted first by filename
        <target_image>   sorted last by filename
        (any number of frames between; we use first vs last)

Category labels (the four buckets your demo set already uses):
    no_change                — expect ZERO anomalies; anything detected is FP
    water_detection          — expect a water/fluid anomaly; oil would be FP
    oil_detection            — expect an oil anomaly; water on its own is FP
    water_and_oil_detection  — expect BOTH oil AND water; missing either is FN

Per-anomaly classification (mutually exclusive: oil takes priority):
    oil   — anomaly_type / description / uncertainty mentions an oil keyword
    water — otherwise, a water/fluid/wet keyword
    other — everything else (does not count toward oil/water scoring)

Outputs:
    <out>/<category>/<case>/inputs/, <category>/<case>/<prompt>/...   (sweep-tree shape)
    <out>/demos_summary.json                                          (full results)
    <out>/demos_scores.json                                           (per-prompt confusion matrices)
    Console: confusion matrix + precision/recall table per prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from change_detection.claude_change_detect import (
    annotate_image,
    assemble_prompt,
    build_composite_image,
    call_claude,
    copy_inputs_to,
    create_run_folder,
    load_prompt,
)

load_dotenv()


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# Ground-truth per category (which presences are expected).
# Short aliases (oil, water, water_oil) and long forms both supported, so
# `dev/demos/.../vlm/oil_detection/` and `dev/demos/cd_demo_2/oil/` both work.
GT: dict[str, dict[str, bool]] = {
    "no_change":               {"oil": False, "water": False},
    "no_change_detection":     {"oil": False, "water": False},
    "water_detection":         {"oil": False, "water": True},
    "water":                   {"oil": False, "water": True},
    "oil_detection":           {"oil": True,  "water": False},
    "oil":                     {"oil": True,  "water": False},
    "water_and_oil_detection": {"oil": True,  "water": True},
    "water_oil":               {"oil": True,  "water": True},
    "oil_and_water":           {"oil": True,  "water": True},
}

# Subdirectory names that count as the per-case references (checked in order).
REF_DIR_NAMES = ("ref", "reference", "refs", "references")

# Oil keywords (letter-class lookarounds, not \b — '_' is a \w char so \b breaks "oil_stain")
OIL_RX = re.compile(
    r"(?<![a-z])(oil|hydrocarbon|petroleum|fuel|diesel|grease|crude|lubricant)(?![a-z])",
    re.IGNORECASE,
)
# Water/fluid keywords — checked AFTER oil so "oil_stain" doesn't double-count.
WATER_RX = re.compile(
    r"(?<![a-z])(water|wet|moisture|puddle|pool(?:ed|ing)?|fluid|liquid|spill|stain|leak|drip|damp|seep)(?![a-z])",
    re.IGNORECASE,
)


def classify_anomaly(a: dict) -> str:
    """Return one of 'oil', 'water', 'other' for a single anomaly."""
    text = " ".join(filter(None, [
        a.get("anomaly_type"), a.get("description"), a.get("uncertainty")
    ]))
    if OIL_RX.search(text):
        return "oil"
    if WATER_RX.search(text):
        return "water"
    return "other"


# ─────────────────────────────────────────────
#  Case discovery
# ─────────────────────────────────────────────

def list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def find_demo_root(start: Path) -> Path:
    """If start has subdirs that match category names, return start.
    Otherwise descend through single-child dirs to find one that does."""
    cur = start
    for _ in range(5):  # bounded
        children = [d for d in cur.iterdir() if d.is_dir()]
        names = {d.name for d in children}
        if any(n in GT for n in names):
            return cur
        if len(children) == 1:
            cur = children[0]
            continue
        # Sometimes there's a "vlm" or "human" sibling — pick the dir containing categories
        for c in children:
            if any((c / k).is_dir() for k in GT):
                return c
        break
    return start  # last resort


def _find_ref_subdir(case_dir: Path) -> Path | None:
    """Return the case's reference subfolder if one exists, else None."""
    for name in REF_DIR_NAMES:
        cand = case_dir / name
        if cand.is_dir():
            return cand
    return None


def discover_cases(root: Path) -> list[dict]:
    """Walk <root>/<category>/<case_dir>/ and return one entry per (case, target).

    Two case layouts are supported:

    A) Per-case ref subfolder (preferred):
           <case>/ref|reference|refs|references/*.{jpg,png,...}    # refs
           <case>/*.{jpg,png,...}                                   # targets (loose)
       Each loose target image becomes a separate test point against the same refs.

    B) Flat case (legacy):
           <case>/*.{jpg,png,...}     # all images here; first sorted = ref, last = target
       One test point per case.

    A case without a ref subfolder AND with <2 loose images is skipped.
    """
    cases: list[dict] = []
    for category in sorted(GT):
        cat_dir = root / category
        if not cat_dir.is_dir():
            continue
        for case_dir in sorted(cat_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            ref_dir = _find_ref_subdir(case_dir)
            loose = list_images(case_dir)  # images directly in case_dir, not in subdirs

            if ref_dir is not None:
                # Layout A — explicit refs
                refs = list_images(ref_dir)
                targets = loose
                if not refs:
                    print(f"[skip] {category}/{case_dir.name}: ref dir is empty.")
                    continue
                if not targets:
                    print(f"[skip] {category}/{case_dir.name}: no target images at case root.")
                    continue
                for tgt in targets:
                    cases.append({
                        "category": category,
                        "case": case_dir.name,
                        "case_dir": case_dir,
                        "ref_dir": ref_dir,
                        "refs": refs,
                        "target": tgt,
                        "n_refs": len(refs),
                        "n_targets": len(targets),
                        "gt": GT[category],
                        "layout": "explicit_refs",
                    })
            else:
                # Layout B — legacy flat
                if len(loose) < 2:
                    continue
                cases.append({
                    "category": category,
                    "case": case_dir.name,
                    "case_dir": case_dir,
                    "ref_dir": None,
                    "refs": [loose[0]],
                    "target": loose[-1],
                    "n_refs": 1,
                    "n_targets": 1,
                    "gt": GT[category],
                    "layout": "flat",
                })
    return cases


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
#  Scoring
# ─────────────────────────────────────────────

def score_case(anomalies: list[dict], gt: dict) -> dict:
    """Return per-dimension TP/FP/FN/TN flags for one case."""
    classes = [classify_anomaly(a) for a in anomalies]
    pred_oil = "oil" in classes
    pred_water = "water" in classes
    out = {"pred_oil": pred_oil, "pred_water": pred_water,
           "n_anomalies": len(anomalies),
           "oil_anomaly_count": sum(1 for c in classes if c == "oil"),
           "water_anomaly_count": sum(1 for c in classes if c == "water"),
           "other_anomaly_count": sum(1 for c in classes if c == "other")}
    for dim in ("oil", "water"):
        truth = gt[dim]
        pred = out[f"pred_{dim}"]
        if truth and pred:        out[f"{dim}_status"] = "TP"
        elif truth and not pred:  out[f"{dim}_status"] = "FN"
        elif not truth and pred:  out[f"{dim}_status"] = "FP"
        else:                     out[f"{dim}_status"] = "TN"
    return out


# ─────────────────────────────────────────────
#  Aggregation / pretty print
# ─────────────────────────────────────────────

def pr(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall    = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def fmt_pct(v: float | None) -> str:
    return f"{v*100:>5.1f}%" if v is not None else "   —  "


def print_scores(by_prompt: dict[str, dict]) -> None:
    print()
    print("===== Per-prompt confusion + precision/recall =====")
    hdr = (f"{'Prompt':<22}  "
           f"{'OilTP':>5} {'OilFP':>5} {'OilFN':>5} {'OilTN':>5}  "
           f"{'OilP':>6} {'OilR':>6}  "
           f"{'WatTP':>5} {'WatFP':>5} {'WatFN':>5} {'WatTN':>5}  "
           f"{'WatP':>6} {'WatR':>6}")
    print(hdr)
    print("-" * len(hdr))
    for prompt, m in by_prompt.items():
        op, orec, _ = pr(m["oil_TP"],   m["oil_FP"],   m["oil_FN"])
        wp, wrec, _ = pr(m["water_TP"], m["water_FP"], m["water_FN"])
        print(f"{prompt:<22}  "
              f"{m['oil_TP']:>5} {m['oil_FP']:>5} {m['oil_FN']:>5} {m['oil_TN']:>5}  "
              f"{fmt_pct(op):>6} {fmt_pct(orec):>6}  "
              f"{m['water_TP']:>5} {m['water_FP']:>5} {m['water_FN']:>5} {m['water_TN']:>5}  "
              f"{fmt_pct(wp):>6} {fmt_pct(wrec):>6}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run prompts against labeled demo cases and score against ground truth.",
    )
    ap.add_argument("--root", default="dev/demos",
                    help="Root containing the demo categories (or a parent of them).")
    ap.add_argument("--out", default="dev/demos_test",
                    help="Output root.")
    ap.add_argument("--prompts-dir", default="src/change_detection/prompts",
                    help="Directory of prompt .txt files.")
    ap.add_argument("--prompts",
                    default="03,04,05",
                    help="Comma-separated prompt stems (prefix match allowed). "
                         "Default: 03,04,05 → 03_focused, 04_calibrated, 05_rigorous.")
    ap.add_argument("--category", default=None,
                    help="Run only cases under this category (e.g. 'oil' or 'water_detection').")
    ap.add_argument("--case", default=None,
                    help="Run only cases whose '<category>/<case>' path contains this substring. "
                         "Examples: 'oil/case_02b', 'water_oil', 'case_01'.")
    ap.add_argument("--merge-input", action="store_true",
                    help="Compose refs + target into a single labeled image and "
                         "send only that to the API. Useful when an upstream "
                         "tool (e.g. Roboflow) restricts uploads to one image.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show the plan, do not call the API.")
    ap.add_argument("--api-key", "-k", default=None)
    args = ap.parse_args()

    start = Path(args.root)
    if not start.is_dir():
        sys.exit(f"Error: root '{start}' is not a directory.")
    root = find_demo_root(start)
    if root != start:
        print(f"(Descended into {root.relative_to(start)} — that's where the category folders live.)")

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

    cases = discover_cases(root)
    if not cases:
        sys.exit(f"No cases discovered under {root}.")

    # Apply --category / --case filters
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.case:
        needle = args.case.replace("\\", "/")
        cases = [c for c in cases
                 if needle in f"{c['category']}/{c['case']}"]
    if not cases:
        sys.exit("No cases matched the --category / --case filters.")

    by_cat: dict[str, int] = defaultdict(int)
    for c in cases: by_cat[c["category"]] += 1

    total_calls = len(cases) * len(prompt_paths)
    print(f"Demo benchmark plan:")
    print(f"  root      : {root}")
    print(f"  cases     : {len(cases)}  ({dict(by_cat)})")
    if args.category or args.case:
        print(f"  filters   : category={args.category!r} case={args.case!r}")
    print(f"  prompts   : {[p.stem for p in prompt_paths]}")
    print(f"  per case  : N refs + 1 target (refs from case/ref dir if present, else legacy)")
    print(f"  calls     : {total_calls}")
    print(f"  output    : {out_root}")
    print()
    if args.dry_run:
        for c in cases:
            ref_src = c["ref_dir"].name + "/" if c["ref_dir"] else "(flat)"
            ref_names = ",".join(p.name for p in c["refs"][:3])
            if len(c["refs"]) > 3: ref_names += ",…"
            print(f"  - {c['category']:<26} {c['case']:<35}  "
                  f"{c['n_refs']} ref(s) from {ref_src}  tgt={c['target'].name}")
            print(f"      refs: {ref_names}")
        print("\nDry run — exiting before any API calls.")
        return

    rows: list[dict] = []
    by_prompt = defaultdict(lambda: {
        "oil_TP": 0, "oil_FP": 0, "oil_FN": 0, "oil_TN": 0,
        "water_TP": 0, "water_FP": 0, "water_FN": 0, "water_TN": 0,
        "tokens": 0, "ran": 0,
    })

    done = 0
    for case in cases:
        # Per-target output dir. When a case has multiple loose targets, each
        # target gets its own subdir so they don't collide.
        target_stem = Path(case["target"]).stem
        if case["n_targets"] > 1:
            case_dir_out = out_root / case["category"] / case["case"] / target_stem
        else:
            case_dir_out = out_root / case["category"] / case["case"]
        case_dir_out.mkdir(parents=True, exist_ok=True)
        refs = [str(p) for p in case["refs"]]
        target = str(case["target"])
        copy_inputs_to(case_dir_out, refs, [], target)

        # In merge-input mode: build one composite per case (shared across all
        # prompts) and remember the offset so bboxes (reported in target-panel
        # coords) annotate the composite at the right place.
        composite_path: str | None = None
        target_offset: tuple[int, int] = (0, 0)
        target_dims: tuple[int, int] | None = None
        if args.merge_input:
            composite_path = str(case_dir_out / "inputs" / "composite.jpg")
            _, target_offset, target_dims = build_composite_image(
                refs, target, composite_path,
            )

        for pp in prompt_paths:
            done += 1
            label = f"{case['category']}__{case['case']}__{target_stem}__{pp.stem}"
            print(f"[{done}/{total_calls}] {label} ...", end=" ", flush=True)
            try:
                base = load_prompt(str(pp))
                if args.merge_input:
                    final = assemble_prompt(base, refs, [], target,
                                            composite_dims=target_dims)
                    api_images = [composite_path]
                else:
                    final = assemble_prompt(base, refs, [], target)
                    api_images = refs + [target]
                result, usage = call_claude(api_images, api_key, final)
                anomalies = result.get("anomalies", [])
                score = score_case(anomalies, case["gt"])

                run_dir = case_dir_out / pp.stem
                run_dir.mkdir(parents=True, exist_ok=True)
                annotated = run_dir / "annotated.jpg"
                if anomalies:
                    if args.merge_input:
                        out_img = annotate_image(composite_path, anomalies,
                                                 str(annotated),
                                                 bbox_offset=target_offset)
                    else:
                        out_img = annotate_image(target, anomalies, str(annotated))
                else:
                    out_img = ""

                create_run_folder(
                    runs_dir=str(case_dir_out),
                    label=label,
                    ref_paths=refs, examples=[], current_path=target,
                    prompt=final, anomalies=anomalies, usage=usage,
                    annotated_image_path=str(annotated) if out_img else None,
                    copy_inputs=False,
                    folder_name=pp.stem,
                    external_inputs_dir=case_dir_out / "inputs",
                )

                # Update confusion-matrix tallies
                s = by_prompt[pp.stem]
                s["ran"] += 1
                s["tokens"] += usage.get("total_tokens") or 0
                s[f"oil_{score['oil_status']}"]   += 1
                s[f"water_{score['water_status']}"] += 1

                rows.append({
                    "category": case["category"],
                    "case": case["case"],
                    "target": Path(target).name,
                    "prompt": pp.stem,
                    "n_anomalies": len(anomalies),
                    "oil_status": score["oil_status"],
                    "water_status": score["water_status"],
                    "oil_anomaly_count": score["oil_anomaly_count"],
                    "water_anomaly_count": score["water_anomaly_count"],
                    "other_anomaly_count": score["other_anomaly_count"],
                    "total_tokens": usage.get("total_tokens"),
                    "run_dir": str(run_dir),
                })
                marker = (("⚑oil-FP " if score["oil_status"] == "FP" else "")
                          + ("⚑oil-FN " if score["oil_status"] == "FN" else "")
                          + ("⚑wat-FP " if score["water_status"] == "FP" else "")
                          + ("⚑wat-FN " if score["water_status"] == "FN" else ""))
                print(f"OK ({len(anomalies)} anom, "
                      f"oil={score['oil_status']}, water={score['water_status']}, "
                      f"{usage.get('total_tokens')} tok) {marker}")
            except Exception as e:
                print(f"FAIL ({type(e).__name__}: {e})")
                traceback.print_exc()
                rows.append({
                    "category": case["category"], "case": case["case"],
                    "prompt": pp.stem,
                    "error": f"{type(e).__name__}: {e}",
                })

    (out_root / "demos_summary.json").write_text(
        json.dumps({"rows": rows, "by_prompt": dict(by_prompt)}, indent=2),
        encoding="utf-8"
    )
    (out_root / "demos_scores.json").write_text(
        json.dumps(dict(by_prompt), indent=2),
        encoding="utf-8"
    )

    print_scores(by_prompt)
    print()
    print(f"Per-case rows  : {out_root / 'demos_summary.json'}")
    print(f"Per-prompt CM  : {out_root / 'demos_scores.json'}")


if __name__ == "__main__":
    main()
