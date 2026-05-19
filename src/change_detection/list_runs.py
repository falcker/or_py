#!/usr/bin/env python3
"""
List and compare runs produced by claude_change_detect.py
==========================================================
Walks a runs directory, reads each run.json, and prints a table.

Usage:
    python list_runs.py                          # list all runs in ./runs
    python list_runs.py --runs-dir path/to/runs  # custom dir
    python list_runs.py --last 10                # most recent 10 only
    python list_runs.py --label-contains water   # filter by label substring
    python list_runs.py --sort tokens            # sort by total_tokens
    python list_runs.py --csv > runs.csv         # CSV output instead of table
    python list_runs.py --diff run1_id run2_id   # compare two runs in detail
"""

import argparse
import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────

def load_runs(runs_dir: Path) -> list[dict]:
    """Load all run.json manifests from subdirectories of runs_dir."""
    if not runs_dir.exists():
        print(f"Error: runs directory not found: {runs_dir}")
        sys.exit(1)

    runs = []
    for subdir in sorted(runs_dir.iterdir()):
        if not subdir.is_dir():
            continue
        manifest_path = subdir / "run.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["_dir"] = subdir
            runs.append(manifest)
        except json.JSONDecodeError:
            print(f"⚠  Skipping {subdir.name}: malformed run.json")
    return runs


def summarize_run(run: dict) -> dict:
    """Pull the fields used in the table from a full manifest."""
    usage = run.get("usage") or {}
    inputs = run.get("inputs") or {}
    anomalies = run.get("anomalies") or []

    anomaly_types: dict[str, int] = {}
    for a in anomalies:
        t = a.get("anomaly_type", "unknown")
        anomaly_types[t] = anomaly_types.get(t, 0) + 1

    return {
        "run_id": run.get("run_id", run["_dir"].name),
        "timestamp": run.get("timestamp", ""),
        "label": run.get("label") or "",
        "prompt_hash": run.get("prompt_hash", "")[:12],
        "model": usage.get("model", ""),
        "n_refs": len(inputs.get("refs") or []),
        "n_examples": len(inputs.get("examples") or []),
        "anomaly_count": len(anomalies),
        "anomaly_types": anomaly_types,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


# ─────────────────────────────────────────────
#  Output formats
# ─────────────────────────────────────────────

def format_anomaly_types(d: dict) -> str:
    """Render {'water_pooling': 2, 'corrosion': 1} as 'water_pooling×2, corrosion×1'."""
    if not d:
        return "-"
    return ", ".join(f"{k}×{v}" for k, v in d.items())


def print_table(summaries: list[dict]) -> None:
    if not summaries:
        print("No runs found.")
        return

    # Columns: timestamp, label, prompt_hash, refs/examples, anomalies, types, tokens
    headers = ["Timestamp", "Label", "Prompt", "R/E", "Anom", "Types", "In", "Out", "Total"]
    rows = []
    for s in summaries:
        rows.append([
            s["timestamp"],
            (s["label"] or "")[:24],
            s["prompt_hash"][:8],
            f"{s['n_refs']}/{s['n_examples']}",
            str(s["anomaly_count"]),
            format_anomaly_types(s["anomaly_types"])[:30],
            f"{s['input_tokens']:,}",
            f"{s['output_tokens']:,}",
            f"{s['total_tokens']:,}",
        ])

    # Calculate widths
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    sep = "─" * (sum(widths) + 2 * (len(widths) - 1))
    print(fmt_row(headers))
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print()
    print(f"  {len(rows)} run(s).  R/E = references / examples.")


def print_csv(summaries: list[dict]) -> None:
    import csv
    writer = csv.writer(sys.stdout)
    writer.writerow([
        "timestamp", "label", "prompt_hash", "model",
        "n_refs", "n_examples", "anomaly_count", "anomaly_types",
        "input_tokens", "output_tokens", "total_tokens",
    ])
    for s in summaries:
        writer.writerow([
            s["timestamp"], s["label"], s["prompt_hash"], s["model"],
            s["n_refs"], s["n_examples"], s["anomaly_count"],
            format_anomaly_types(s["anomaly_types"]),
            s["input_tokens"], s["output_tokens"], s["total_tokens"],
        ])


# ─────────────────────────────────────────────
#  Diff
# ─────────────────────────────────────────────

def find_run(runs: list[dict], run_id: str) -> dict | None:
    """Find a run by exact id, prefix match, or label substring."""
    # Exact id
    for r in runs:
        if r.get("run_id") == run_id:
            return r
    # Prefix on timestamp
    matches = [r for r in runs if r.get("run_id", "").startswith(run_id)]
    if len(matches) == 1:
        return matches[0]
    # Label substring
    matches = [r for r in runs if run_id.lower() in (r.get("label") or "").lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"⚠  Ambiguous identifier '{run_id}' — matches:")
        for m in matches:
            print(f"     {m.get('run_id')}  ({m.get('label')})")
        return None
    return None


def print_diff(run_a: dict, run_b: dict) -> None:
    sa = summarize_run(run_a)
    sb = summarize_run(run_b)

    print(f"\nComparing two runs:")
    print(f"  A:  {sa['run_id']}  ({sa['label'] or 'no label'})")
    print(f"  B:  {sb['run_id']}  ({sb['label'] or 'no label'})")
    print()

    fields = [
        ("Prompt hash", "prompt_hash"),
        ("Model", "model"),
        ("Refs / Examples", lambda s: f"{s['n_refs']} / {s['n_examples']}"),
        ("Anomaly count", "anomaly_count"),
        ("Anomaly types", lambda s: format_anomaly_types(s["anomaly_types"])),
        ("Input tokens", lambda s: f"{s['input_tokens']:,}"),
        ("Output tokens", lambda s: f"{s['output_tokens']:,}"),
        ("Total tokens", lambda s: f"{s['total_tokens']:,}"),
    ]

    label_w = max(len(name) for name, _ in fields)
    val_w_a = 30
    print(f"  {'Field'.ljust(label_w)}  {'A'.ljust(val_w_a)}  B")
    print(f"  {'-' * label_w}  {'-' * val_w_a}  {'-' * 30}")

    for name, getter in fields:
        va = getter(sa) if callable(getter) else sa.get(getter, "")
        vb = getter(sb) if callable(getter) else sb.get(getter, "")
        marker = "  " if str(va) == str(vb) else "≠ "
        print(f"  {name.ljust(label_w)}  {str(va).ljust(val_w_a)}  {marker}{vb}")

    # Detailed anomaly diff
    print()
    print("Anomaly details:")
    anoms_a = run_a.get("anomalies") or []
    anoms_b = run_b.get("anomalies") or []
    for label, anoms in [("A", anoms_a), ("B", anoms_b)]:
        print(f"\n  {label} ({len(anoms)}):")
        if not anoms:
            print("    (none)")
        for i, a in enumerate(anoms, 1):
            t = a.get("anomaly_type", "?")
            sev = a.get("severity", "?")
            conf = a.get("confidence")
            conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else "?"
            print(f"    [{i}] {t}  severity={sev}  confidence={conf_str}")
            desc = a.get("description", "")
            if desc:
                print(f"        {desc[:80]}")

    # Hint at where prompt diff lives
    if sa["prompt_hash"] != sb["prompt_hash"]:
        print()
        print("Prompts differ. To compare:")
        print(f"  diff {run_a['_dir'] / 'prompt.txt'} \\")
        print(f"       {run_b['_dir'] / 'prompt.txt'}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="List and compare claude_change_detect runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs-dir", default="runs", help="Directory of run folders (default: runs)")
    parser.add_argument("--last", type=int, default=None, help="Show only the N most recent runs")
    parser.add_argument("--label-contains", default=None, help="Filter by label substring")
    parser.add_argument(
        "--sort", choices=["time", "tokens", "anomalies", "label"],
        default="time", help="Sort order (default: time)",
    )
    parser.add_argument("--csv", action="store_true", help="Output as CSV instead of a table")
    parser.add_argument(
        "--diff", nargs=2, metavar=("RUN_A", "RUN_B"),
        help="Compare two runs in detail. Each arg can be a run_id, timestamp prefix, or label substring.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    runs = load_runs(runs_dir)

    if args.diff:
        run_a = find_run(runs, args.diff[0])
        run_b = find_run(runs, args.diff[1])
        if not run_a or not run_b:
            if not run_a:
                print(f"Error: could not find run '{args.diff[0]}'")
            if not run_b:
                print(f"Error: could not find run '{args.diff[1]}'")
            sys.exit(1)
        print_diff(run_a, run_b)
        return

    # Filter
    if args.label_contains:
        runs = [r for r in runs if args.label_contains.lower() in (r.get("label") or "").lower()]

    summaries = [summarize_run(r) for r in runs]

    # Sort
    if args.sort == "time":
        summaries.sort(key=lambda s: s["timestamp"])
    elif args.sort == "tokens":
        summaries.sort(key=lambda s: s["total_tokens"], reverse=True)
    elif args.sort == "anomalies":
        summaries.sort(key=lambda s: s["anomaly_count"], reverse=True)
    elif args.sort == "label":
        summaries.sort(key=lambda s: s["label"])

    # Limit
    if args.last:
        summaries = summaries[-args.last:] if args.sort == "time" else summaries[:args.last]

    if args.csv:
        print_csv(summaries)
    else:
        print_table(summaries)


if __name__ == "__main__":
    main()
