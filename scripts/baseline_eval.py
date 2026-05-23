from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    ensure_utf8_stdio,
    evaluate_standardized_records,
    load_model_and_tokenizer,
    load_records_auto,
    resolve_input_path,
    resolve_output_path,
    save_json,
    timestamp,
    to_jsonable,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline evaluation for a knowledge editing dataset.")
    parser.add_argument("--data_path", type=str, default="custom_10.json")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="results/baseline")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", "--torch_dtype", dest="dtype", type=str, default="auto")
    return parser


def main() -> None:
    ensure_utf8_stdio()
    args = build_argparser().parse_args()
    data_path = resolve_input_path(args.data_path)
    output_dir = resolve_output_path(args.output_dir)
    records = load_records_auto(data_path)

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
    )

    evaluated_records, summary = evaluate_standardized_records(
        model=model,
        tokenizer=tokenizer,
        records=records,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    result = {
        "task": "baseline",
        "model_name": args.model_name,
        "data_path": str(data_path),
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "summary": summary,
        "records": evaluated_records,
    }

    output_path = output_dir / f"baseline_{timestamp()}.json"
    save_json(to_jsonable(result), output_path)

    print(f"Saved baseline results to: {output_path}")
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
