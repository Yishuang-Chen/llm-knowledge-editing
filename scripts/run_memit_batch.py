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
    sample_records,
    save_json,
    summarize_stage_records,
    timestamp,
    to_jsonable,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MEMIT batch editing on a KnowEdit-style dataset.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--hparams_path", type=str, default="configs/memit_qwen2.5_0.5b.yaml")
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="results/memit")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8, help="Generation/evaluation batch size.")
    parser.add_argument(
        "--edit_batch_size",
        type=int,
        default=0,
        help="Number of edits injected in one MEMIT batch. 0 means use all sampled records in one batch.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", "--torch_dtype", dest="dtype", type=str, default="auto")
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
    if len(records) < args.sample_size:
        raise ValueError(
            f"Only {len(records)} valid records remain after filtering, fewer than sample_size={args.sample_size}."
        )
    records = sample_records(records, sample_size=args.sample_size, seed=args.seed)

    ensure_easyedit_on_sys_path()
    try:
        from easyeditor import BaseEditor, MEMITHyperParams
    except ImportError as exc:
        raise ImportError(
            "Failed to import EasyEdit. Activate the correct conda environment and install EasyEdit first."
        ) from exc

    print("Running pre-edit evaluation...")
    pre_eval_start = time.perf_counter()
    pre_model, pre_tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
    )
    pre_records, _ = evaluate_standardized_records(
        model=pre_model,
        tokenizer=pre_tokenizer,
        records=records,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        progress_prefix="pre-eval",
    )
    pre_eval_time = time.perf_counter() - pre_eval_start
    print(f"Pre-edit evaluation finished in {pre_eval_time:.2f}s")
    clear_memory(pre_model)

    hparams = MEMITHyperParams.from_hparams(str(hparams_path))
    edit_batch_size = len(records) if args.edit_batch_size <= 0 else min(args.edit_batch_size, len(records))
    hparams.batch_size = edit_batch_size
    editor = BaseEditor.from_hparams(hparams)

    prompts = [record["prompt"] for record in records]
    ground_truth = [first_answer(record.get("ground_truth")) for record in records]
    target_new = [first_answer(record.get("target_new")) for record in records]
    rephrase_prompts = [record["rephrase_prompt"] for record in records]
    locality_inputs = build_locality_inputs(records)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("Running MEMIT batch editing...")
    t0 = time.perf_counter()
    metrics, edited_model, _ = editor.batch_edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=[record["subject"] for record in records],
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        sequential_edit=True,
    )
    edit_time = time.perf_counter() - t0

    peak_memory_gb = None
    if args.device.startswith("cuda") and torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("Running post-edit evaluation...")
    post_eval_start = time.perf_counter()
    post_tokenizer = get_editor_tokenizer(editor, pre_tokenizer)
    post_records, _ = evaluate_standardized_records(
        model=edited_model,
        tokenizer=post_tokenizer,
        records=records,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        progress_prefix="post-eval",
    )
    post_eval_time = time.perf_counter() - post_eval_start
    print(f"Post-edit evaluation finished in {post_eval_time:.2f}s")

    merged_records: list[dict] = []
    for idx, (record, pre_record, post_record) in enumerate(zip(records, pre_records, post_records)):
        merged_records.append(
            {
                "source_index": record["source_index"],
                "subject": record.get("subject"),
                "easyedit_metrics_raw": metrics[idx] if isinstance(metrics, list) and idx < len(metrics) else None,
                "pre": pre_record,
                "post": post_record,
            }
        )

    clear_memory(edited_model, editor)

    summary_pre = summarize_stage_records(merged_records, "pre")
    summary_post = summarize_stage_records(merged_records, "post")
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
        "task": "memit_batch",
        "method": "MEMIT",
        "model_name": model_name,
        "hparams_path": str(hparams_path.resolve()),
        "data_path": str(data_path),
        "device": args.device,
        "dtype": args.dtype,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "edit_batch_size": edit_batch_size,
        "summary": {
            "pre": summary_pre,
            "post": summary_post,
            "delta": summary_delta,
            "pre_eval_time_seconds": pre_eval_time,
            "edit_time_seconds": edit_time,
            "post_eval_time_seconds": post_eval_time,
            "peak_memory_gb": peak_memory_gb,
        },
        "easyedit_metrics_raw": metrics,
        "records": merged_records,
    }

    output_path = output_dir / f"memit_batch_{timestamp()}.json"
    save_json(to_jsonable(result), output_path)

    print(f"Saved MEMIT results to: {output_path}")
    print("Pre summary:", summary_pre)
    print("Post summary:", summary_post)
    print("Delta:", summary_delta)
    print(f"Pre-eval time (s): {pre_eval_time:.4f}")
    print(f"Edit time (s): {edit_time:.4f}")
    print(f"Post-eval time (s): {post_eval_time:.4f}")
    print(f"Peak memory (GB): {peak_memory_gb}")


if __name__ == "__main__":
    main()
