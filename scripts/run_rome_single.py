from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common import (
    build_locality_inputs,
    clear_memory,
    ensure_easyedit_on_sys_path,
    ensure_utf8_stdio,
    evaluate_standardized_records,
    filter_records_for_editing,
    first_answer,
    load_model_and_tokenizer,
    load_records_auto,
    load_yaml,
    resolve_hparams_path,
    resolve_input_path,
    resolve_output_path,
    save_json,
    summarize_stage_records,
    timestamp,
    to_jsonable,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ROME single-fact editing on a custom dataset.")
    parser.add_argument("--data_path", type=str, default="custom_10.json")
    parser.add_argument("--hparams_path", type=str, default="configs/rome_qwen2.5_0.5b.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="results/rome")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", "--torch_dtype", dest="dtype", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Debug option. 0 means use all records.")
    return parser


def get_editor_tokenizer(editor, fallback_tokenizer):
    return getattr(editor, "tok", None) or getattr(editor, "tokenizer", None) or fallback_tokenizer


def main() -> None:
    ensure_utf8_stdio()
    args = build_argparser().parse_args()

    hparams_path = resolve_hparams_path(args.hparams_path)
    data_path = resolve_input_path(args.data_path)
    output_dir = resolve_output_path(args.output_dir)
    hparams_dict = load_yaml(hparams_path)
    model_name = hparams_dict["model_name"]

    records = load_records_auto(data_path)
    records = filter_records_for_editing(records)
    if args.limit > 0:
        records = records[: args.limit]

    if not records:
        raise ValueError("No valid records left after filtering. Check rephrase/locality fields.")

    ensure_easyedit_on_sys_path()
    try:
        from easyeditor import BaseEditor, ROMEHyperParams
    except ImportError as exc:
        raise ImportError(
            "Failed to import EasyEdit. Activate the correct conda environment and install EasyEdit first."
        ) from exc

    hparams = ROMEHyperParams.from_hparams(str(hparams_path))
    all_results: list[dict] = []

    for idx, record in enumerate(records, start=1):
        print(f"[{idx}/{len(records)}] Processing subject: {record.get('subject')}")

        pre_model, pre_tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
        )
        pre_records, _ = evaluate_standardized_records(
            model=pre_model,
            tokenizer=pre_tokenizer,
            records=[record],
            max_new_tokens=args.max_new_tokens,
            batch_size=1,
        )
        clear_memory(pre_model)

        editor = BaseEditor.from_hparams(hparams)

        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        metrics, edited_model, _ = editor.edit(
            prompts=[record["prompt"]],
            ground_truth=[first_answer(record.get("ground_truth"))],
            target_new=[first_answer(record.get("target_new"))],
            subject=[record["subject"]],
            rephrase_prompts=[record["rephrase_prompt"]],
            locality_inputs=build_locality_inputs([record]),
            sequential_edit=False,
        )
        edit_time = time.perf_counter() - t0

        peak_memory_gb = None
        if args.device.startswith("cuda") and torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        post_tokenizer = get_editor_tokenizer(editor, pre_tokenizer)
        post_records, _ = evaluate_standardized_records(
            model=edited_model,
            tokenizer=post_tokenizer,
            records=[record],
            max_new_tokens=args.max_new_tokens,
            batch_size=1,
        )

        result_record = {
            "source_index": record["source_index"],
            "subject": record.get("subject"),
            "edit_time_seconds": edit_time,
            "peak_memory_gb": peak_memory_gb,
            "easyedit_metrics_raw": metrics[0] if isinstance(metrics, list) and metrics else metrics,
            "pre": pre_records[0],
            "post": post_records[0],
        }
        all_results.append(result_record)

        clear_memory(edited_model, editor)

    summary_pre = summarize_stage_records(all_results, "pre")
    summary_post = summarize_stage_records(all_results, "post")
    summary_delta = {
        "efficacy_es_delta": None
        if summary_pre["efficacy_es"] is None or summary_post["efficacy_es"] is None
        else summary_post["efficacy_es"] - summary_pre["efficacy_es"],
        "generalization_ps_delta": None
        if summary_pre["generalization_ps"] is None or summary_post["generalization_ps"] is None
        else summary_post["generalization_ps"] - summary_pre["generalization_ps"],
        "locality_ns_delta": None
        if summary_pre["locality_ns"] is None or summary_post["locality_ns"] is None
        else summary_post["locality_ns"] - summary_pre["locality_ns"],
    }

    result = {
        "task": "rome_single",
        "method": "ROME",
        "model_name": model_name,
        "hparams_path": str(hparams_path.resolve()),
        "data_path": str(data_path),
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "summary": {
            "pre": summary_pre,
            "post": summary_post,
            "delta": summary_delta,
            "edit_time_seconds_mean": sum(r["edit_time_seconds"] for r in all_results) / len(all_results),
            "edit_time_seconds_total": sum(r["edit_time_seconds"] for r in all_results),
            "peak_memory_gb_max": max(
                (r["peak_memory_gb"] for r in all_results if r["peak_memory_gb"] is not None),
                default=None,
            ),
        },
        "records": all_results,
    }

    output_path = output_dir / f"rome_single_{timestamp()}.json"
    save_json(to_jsonable(result), output_path)

    print(f"Saved ROME results to: {output_path}")
    print("Pre summary:", summary_pre)
    print("Post summary:", summary_post)
    print("Delta:", summary_delta)


if __name__ == "__main__":
    main()
