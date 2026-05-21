#!/usr/bin/env python3
"""
Batch-run a set of prompts against a folder tree of test scenarios.

Folder layout (scenario-per-folder):
    <root>/
        <scenario_name>/
            refs/      *.{jpg,jpeg,png,webp,gif}   (>= 1)
            examples/  *.{jpg,jpeg,png,webp,gif}   (optional; filename stem = label)
            targets/   *.{jpg,jpeg,png,webp,gif}   (>= 1)

For each (scenario × target × prompt) the runner writes a single
deterministic folder containing every artifact:

    <out>/<scenario>/<target_stem>/<prompt_stem>/
        annotated.jpg       (only if anomalies were detected)
        anomalies.json
        prompt.txt
        message.txt
        run.json
        inputs/             (copies of refs, examples, target)

Aggregations:
    <out>/<scenario>/<target_stem>/summary.json   per-prompt rollup for one target
    <out>/sweep_summary.json                      all (scenario,target,prompt) rows

Usage:
    python -m change_detection.sweep <root> [options]

Options:
    --prompts-dir DIR   directory of prompt .txt files (default:
                        src/change_detection/prompts)
    --prompts NAMES     comma-separated prompt stems to run (default: all)
    --scenarios NAMES   comma-separated scenario folder names to run
                        (default: all)
    --out DIR           output root (default: dev/sweeps)
    --dry-run           list what would run; do not call the API
    --api-key KEY       overrides ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

from change_detection.claude_change_detect import (
    assemble_prompt,
    call_claude,
    annotate_image,
    create_run_folder,
    load_prompt,
)

load_dotenv()


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


# ─────────────────────────────────────────────
#  Scenario discovery
# ─────────────────────────────────────────────

def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _is_excluded_scenario(name: str) -> bool:
    """Folder names that are deliberately not scenarios.

    Convention: 'template' (any case), or any folder starting with '_' or '.'.
    Used for scaffolding folders the user keeps around for copy-paste but
    doesn't want the sweep to touch.
    """
    if name.lower() == "template":
        return True
    return name.startswith(("_", "."))


def discover_scenarios(root: Path) -> list[dict]:
    """Walk <root> and return scenarios that have at least one ref and one target."""
    scenarios = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if _is_excluded_scenario(child.name):
            continue
        refs = list_images(child / "refs")
        targets = list_images(child / "targets")
        examples_paths = list_images(child / "examples")
        examples = [(p, p.stem) for p in examples_paths]  # filename stem = label

        if not targets:
            print(f"[skip] {child.name}: needs at least one image in targets/ "
                  f"(refs={len(refs)}, examples={len(examples)}, targets=0)")
            continue
        if not refs and not examples:
            print(f"[skip] {child.name}: needs at least one context image "
                  f"(refs or examples). Found targets={len(targets)} only.")
            continue
        if not refs:
            print(f"[note] {child.name}: no refs — comparison prompts (01-06) "
                  f"may underperform; only examples/{len(examples)} as context.")

        scenarios.append({
            "name": child.name,
            "path": child,
            "refs": refs,
            "examples": examples,
            "targets": targets,
        })
    return scenarios


def list_prompts(prompts_dir: Path) -> list[Path]:
    return sorted(p for p in prompts_dir.iterdir()
                  if p.is_file() and p.suffix.lower() == ".txt")


# ─────────────────────────────────────────────
#  One unit of work
# ─────────────────────────────────────────────

def run_one(
    *,
    scenario: dict,
    target: Path,
    prompt_path: Path,
    api_key: str,
    out_root: Path,
) -> dict:
    """Execute one (scenario, target, prompt) combination. Returns a summary dict.

    All artifacts (annotated image, anomalies.json, run.json, prompt.txt,
    message.txt, inputs/) live in a single deterministic folder:
        <out_root>/<scenario>/<target_stem>/<prompt_stem>/
    """
    refs_str = [str(p) for p in scenario["refs"]]
    examples_pairs = [(str(p), label) for p, label in scenario["examples"]]
    target_str = str(target)

    base_prompt = load_prompt(str(prompt_path))
    final_prompt = assemble_prompt(base_prompt, refs_str, examples_pairs, target_str)

    all_paths = refs_str + [p for p, _ in examples_pairs] + [target_str]

    target_dir = out_root / scenario["name"] / target.stem
    run_dir = target_dir / prompt_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = run_dir / "annotated.jpg"

    t0 = time.time()
    result, usage = call_claude(all_paths, api_key, final_prompt)
    elapsed = time.time() - t0
    anomalies = result.get("anomalies", [])

    out_img = annotate_image(target_str, anomalies, str(annotated_path)) if anomalies else ""

    # Write the canonical run folder (prompt.txt, message.txt, anomalies.json,
    # run.json, inputs/). Folder name is the prompt stem — deterministic, so
    # re-runs overwrite cleanly.
    examples_for_log = [
        {"path": p, "type": label, "bbox": None}
        for p, label in examples_pairs
    ]
    create_run_folder(
        runs_dir=str(target_dir),
        label=f"{scenario['name']}__{target.stem}__{prompt_path.stem}",
        ref_paths=refs_str,
        examples=examples_for_log,
        current_path=target_str,
        prompt=final_prompt,
        anomalies=anomalies,
        usage=usage,
        annotated_image_path=str(annotated_path) if out_img else None,
        copy_inputs=True,
        folder_name=prompt_path.stem,
    )

    return {
        "scenario": scenario["name"],
        "target": target.name,
        "prompt": prompt_path.stem,
        "run_dir": str(run_dir),
        "elapsed_seconds": round(elapsed, 2),
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "usage": usage,
        "annotated_image": str(annotated_path) if out_img else None,
    }


# ─────────────────────────────────────────────
#  Per-target aggregation
# ─────────────────────────────────────────────

def write_target_summary(out_dir: Path, results: list[dict]) -> None:
    """One row per prompt for this target."""
    rows = [{
        "prompt": r["prompt"],
        "anomaly_count": r["anomaly_count"],
        "top_confidence": max(
            (a.get("confidence", 0) for a in r["anomalies"] if isinstance(a.get("confidence"), (int, float))),
            default=None,
        ),
        "input_tokens": r["usage"].get("input_tokens"),
        "output_tokens": r["usage"].get("output_tokens"),
        "elapsed_seconds": r["elapsed_seconds"],
    } for r in results]
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch prompt sweep across scenario folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("root", help="Root directory containing scenario subfolders.")
    parser.add_argument("--prompts-dir",
                        default="src/change_detection/prompts",
                        help="Directory of prompt .txt files.")
    parser.add_argument("--prompts", default=None,
                        help="Comma-separated prompt stems to run (default: all).")
    parser.add_argument("--scenarios", default=None,
                        help="Comma-separated scenario folder names to run (default: all).")
    parser.add_argument("--out", default="dev/sweeps",
                        help="Output root (default: dev/sweeps).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would run; do not call the API.")
    parser.add_argument("--api-key", "-k", default=None,
                        help="Anthropic API key (overrides ANTHROPIC_API_KEY).")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"Error: root '{root}' is not a directory.")

    prompts_dir = Path(args.prompts_dir)
    if not prompts_dir.is_dir():
        sys.exit(f"Error: prompts-dir '{prompts_dir}' is not a directory.")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("Error: provide --api-key or set ANTHROPIC_API_KEY.")

    # Discover and filter
    scenarios = discover_scenarios(root)
    if args.scenarios:
        keep = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        scenarios = [s for s in scenarios if s["name"] in keep]

    prompts = list_prompts(prompts_dir)
    if args.prompts:
        keep = {s.strip() for s in args.prompts.split(",") if s.strip()}
        prompts = [p for p in prompts if p.stem in keep]

    if not scenarios:
        sys.exit("No scenarios to run (after filtering).")
    if not prompts:
        sys.exit("No prompts to run (after filtering).")

    total_runs = sum(len(s["targets"]) for s in scenarios) * len(prompts)
    print(f"Sweep plan: {len(scenarios)} scenario(s) × prompts={len(prompts)} "
          f"× targets/scenario varies = {total_runs} API call(s).")
    for s in scenarios:
        print(f"  scenario '{s['name']}': "
              f"{len(s['refs'])} ref, {len(s['examples'])} example, "
              f"{len(s['targets'])} target(s)")
    print(f"  prompts: {', '.join(p.stem for p in prompts)}")
    print(f"  output : {out_root}")
    print()

    if args.dry_run:
        print("Dry run — exiting before any API calls.")
        return

    sweep_summary: list[dict] = []
    done = 0

    for scenario in scenarios:
        for target in scenario["targets"]:
            target_dir = out_root / scenario["name"] / target.stem
            target_dir.mkdir(parents=True, exist_ok=True)
            target_results: list[dict] = []

            for prompt_path in prompts:
                done += 1
                label = f"{scenario['name']}/{target.stem}/{prompt_path.stem}"
                print(f"[{done}/{total_runs}] {label} ...", end=" ", flush=True)
                try:
                    payload = run_one(
                        scenario=scenario,
                        target=target,
                        prompt_path=prompt_path,
                        api_key=api_key,
                        out_root=out_root,
                    )
                    print(f"OK ({payload['anomaly_count']} anomaly, "
                          f"{payload['usage'].get('total_tokens')} tok, "
                          f"{payload['elapsed_seconds']}s)")
                    target_results.append(payload)
                    sweep_summary.append({
                        "scenario": payload["scenario"],
                        "target": payload["target"],
                        "prompt": payload["prompt"],
                        "run_dir": payload["run_dir"],
                        "anomaly_count": payload["anomaly_count"],
                        "total_tokens": payload["usage"].get("total_tokens"),
                        "elapsed_seconds": payload["elapsed_seconds"],
                    })
                except Exception as e:
                    print(f"FAIL ({type(e).__name__}: {e})")
                    traceback.print_exc()
                    sweep_summary.append({
                        "scenario": scenario["name"],
                        "target": target.name,
                        "prompt": prompt_path.stem,
                        "error": f"{type(e).__name__}: {e}",
                    })

            if target_results:
                write_target_summary(target_dir, target_results)

    (out_root / "sweep_summary.json").write_text(
        json.dumps(sweep_summary, indent=2), encoding="utf-8"
    )

    # Final terse report
    print()
    print("===== Sweep complete =====")
    ok = sum(1 for r in sweep_summary if "error" not in r)
    fail = len(sweep_summary) - ok
    detections = sum(r.get("anomaly_count", 0) for r in sweep_summary)
    tokens = sum(r.get("total_tokens", 0) or 0 for r in sweep_summary)
    print(f"  runs       : {len(sweep_summary)} ({ok} ok, {fail} failed)")
    print(f"  detections : {detections} total anomaly entries across all runs")
    print(f"  tokens     : {tokens:,} total")
    print(f"  summary    : {out_root / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
