from __future__ import annotations

import argparse
from pathlib import Path

from common import ensure_utf8_stdio, load_json, resolve_input_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize one or more baseline/edit result JSON files.")
    parser.add_argument("--path", type=str, required=True, help="A result JSON file or a directory of JSON files.")
    return parser


def collect_files(path_str: str) -> list[Path]:
    path = resolve_input_path(path_str)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.json"))
    raise FileNotFoundError(path_str)


def print_summary(result_path: Path) -> None:
    data = load_json(result_path)
    task = data.get("task", "unknown")
    print("=" * 80)
    print(f"File: {result_path}")
    print(f"Task: {task}")

    if task == "baseline":
        summary = data["summary"]
        print(f"Model: {data.get('model_name')}")
        print(f"Samples: {summary.get('n_samples')}")
        print(f"Target hit rate: {summary.get('efficacy_es')}")
        print(f"Rephrase target hit rate: {summary.get('generalization_ps')}")
        print(f"Locality ground-truth hit rate: {summary.get('locality_ns')}")
        print(f"Old fact hit rate: {summary.get('old_fact_hit_rate')}")
        return

    summary = data.get("summary", {})
    pre = summary.get("pre", {})
    post = summary.get("post", {})
    delta = summary.get("delta", {})

    print(f"Method: {data.get('method')}")
    print(f"Model: {data.get('model_name')}")
    print(f"Pre ES: {pre.get('efficacy_es')}")
    print(f"Post ES: {post.get('efficacy_es')}")
    print(f"Pre PS: {pre.get('generalization_ps')}")
    print(f"Post PS: {post.get('generalization_ps')}")
    print(f"Pre NS: {pre.get('locality_ns')}")
    print(f"Post NS: {post.get('locality_ns')}")
    print(f"Delta ES: {delta.get('efficacy_es_delta')}")
    print(f"Delta PS: {delta.get('generalization_ps_delta')}")
    print(f"Delta NS: {delta.get('locality_ns_delta')}")
    print(f"Extra summary: {summary}")


def main() -> None:
    ensure_utf8_stdio()
    args = build_argparser().parse_args()
    files = collect_files(args.path)
    if not files:
        raise FileNotFoundError(f"No JSON files found under: {args.path}")
    for file_path in files:
        print_summary(file_path)


if __name__ == "__main__":
    main()
