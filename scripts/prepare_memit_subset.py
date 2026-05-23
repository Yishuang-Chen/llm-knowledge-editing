from __future__ import annotations

import argparse

from common import (
    ensure_utf8_stdio,
    filter_records_for_editing,
    load_records_auto,
    resolve_input_path,
    resolve_output_path,
    sample_records,
    save_json,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a filtered/sampled MEMIT dataset subset from a full JSON dataset."
    )
    parser.add_argument("--input_path", type=str, required=True, help="Full dataset JSON path.")
    parser.add_argument("--output_path", type=str, required=True, help="Output subset JSON path.")
    parser.add_argument("--sample_size", type=int, required=True, help="Number of records to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    return parser


def main() -> None:
    ensure_utf8_stdio()
    args = build_argparser().parse_args()

    input_path = resolve_input_path(args.input_path)
    output_path = resolve_output_path(args.output_path)

    records = load_records_auto(input_path)
    total_records = len(records)

    filtered_records = filter_records_for_editing(records)
    filtered_count = len(filtered_records)

    if filtered_count < args.sample_size:
        raise ValueError(
            f"Only {filtered_count} valid records remain after filtering, fewer than sample_size={args.sample_size}."
        )

    sampled_records = sample_records(filtered_records, sample_size=args.sample_size, seed=args.seed)
    save_json(sampled_records, output_path)

    print(f"Input file: {input_path}")
    print(f"Total records: {total_records}")
    print(f"Valid records after filtering: {filtered_count}")
    print(f"Saved sampled subset to: {output_path}")
    print(f"Sample size: {len(sampled_records)}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
