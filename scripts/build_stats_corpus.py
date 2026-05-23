from __future__ import annotations

import argparse
import json

from common import ensure_utf8_stdio, load_json, resolve_input_path, resolve_output_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a local text corpus JSONL for MEMIT/ROME covariance statistics."
    )
    parser.add_argument("--input_path", type=str, required=True, help="Source dataset JSON path.")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL path with {'text': ...}.")
    return parser


def iter_texts(record: dict) -> list[str]:
    texts: list[str] = []

    for key in ("prompt", "rephrase_prompt"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())

    locality = record.get("locality")
    if isinstance(locality, dict):
        for _, items in locality.items():
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    prompt = item.get("prompt")
                    if isinstance(prompt, str) and prompt.strip():
                        texts.append(prompt.strip())

    portability = record.get("portability")
    if isinstance(portability, dict):
        for _, items in portability.items():
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    prompt = item.get("prompt")
                    if isinstance(prompt, str) and prompt.strip():
                        texts.append(prompt.strip())

    return texts


def main() -> None:
    ensure_utf8_stdio()
    args = build_argparser().parse_args()

    input_path = resolve_input_path(args.input_path)
    output_path = resolve_output_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_json(input_path)
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list of records.")

    count = 0
    seen: set[str] = set()
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            if not isinstance(record, dict):
                continue
            for text in iter_texts(record):
                if text in seen:
                    continue
                seen.add(text)
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Unique text rows written: {count}")


if __name__ == "__main__":
    main()
