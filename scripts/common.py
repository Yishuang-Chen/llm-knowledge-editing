from __future__ import annotations

import gc
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EASYEDIT_REPO_ROOT = PROJECT_ROOT / "EasyEdit"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict):
        raise ValueError(f"YAML content must be a dict: {path}")
    return content


def resolve_input_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
        candidates.append(SCRIPT_DIR / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Cannot find input path: {path_str}")


def resolve_output_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def ensure_easyedit_on_sys_path() -> Path:
    repo_root = EASYEDIT_REPO_ROOT
    package_dir = repo_root / "easyeditor"
    if not package_dir.exists():
        raise FileNotFoundError(
            f"EasyEdit repository not found at expected path: {repo_root}. "
            "Clone EasyEdit into the project root first."
        )
    repo_root_str = str(repo_root.resolve())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root.resolve()


def ensure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def resolve_hparams_path(path_str: str) -> Path:
    path = Path(path_str)
    candidates = [path]
    if path.suffix == "":
        candidates.append(path.with_suffix(".yaml"))
    if not path.is_absolute():
        project_candidate = PROJECT_ROOT / path
        candidates.append(project_candidate)
        if path.suffix == "":
            candidates.append(project_candidate.with_suffix(".yaml"))
        script_candidate = SCRIPT_DIR / path
        candidates.append(script_candidate)
        if path.suffix == "":
            candidates.append(script_candidate.with_suffix(".yaml"))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find hparams file: {path_str}")


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def flatten_answers(value: Any) -> list[str]:
    answers: list[str] = []
    if value is None:
        return answers
    if isinstance(value, str):
        text = value.strip()
        if text:
            answers.append(text)
        return answers
    if isinstance(value, (list, tuple, set)):
        for item in value:
            answers.extend(flatten_answers(item))
        return answers
    text = str(value).strip()
    if text:
        answers.append(text)
    return answers


def first_answer(value: Any) -> str | None:
    answers = flatten_answers(value)
    return answers[0] if answers else None


LOCALITY_FIELD_NAME_MAP = {
    "locality_rs": "Relation_Specificity",
    "locality_f": "Forgetfulness",
}


def _extract_locality_groups_from_record(record: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    locality_groups: dict[str, dict[str, list[str]]] = {}

    locality = record.get("locality")
    if isinstance(locality, dict):
        for locality_key, items in locality.items():
            if not items:
                continue

            # Already-normalized format:
            # {
            #   "Relation_Specificity": {
            #       "prompt": [...],
            #       "ground_truth": [...]
            #   }
            # }
            if isinstance(items, dict) and "prompt" in items and "ground_truth" in items:
                prompts_raw = items.get("prompt")
                ground_truths_raw = items.get("ground_truth")

                prompts = flatten_answers(prompts_raw)
                ground_truths = flatten_answers(ground_truths_raw)

                if prompts and ground_truths:
                    locality_groups[locality_key] = {
                        "prompt": prompts,
                        "ground_truth": ground_truths,
                    }
                continue

            if isinstance(items, dict):
                items = [items]

            prompts: list[str] = []
            ground_truths: list[str] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                prompt = item.get("prompt")
                ground_truth = first_answer(item.get("ground_truth"))
                if prompt is None or ground_truth is None:
                    continue
                prompts.append(str(prompt))
                ground_truths.append(ground_truth)

            if prompts:
                locality_groups[locality_key] = {
                    "prompt": prompts,
                    "ground_truth": ground_truths,
                }

    if locality_groups:
        return locality_groups

    for field_name, locality_key in LOCALITY_FIELD_NAME_MAP.items():
        items = record.get(field_name)
        if not items:
            continue
        if isinstance(items, dict):
            items = [items]

        prompts: list[str] = []
        ground_truths: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            prompt = item.get("prompt")
            ground_truth = first_answer(item.get("ground_truth"))
            if prompt is None or ground_truth is None:
                continue
            prompts.append(str(prompt))
            ground_truths.append(ground_truth)

        if prompts:
            locality_groups[locality_key] = {
                "prompt": prompts,
                "ground_truth": ground_truths,
            }

    if locality_groups:
        return locality_groups

    locality_prompt = record.get("locality_prompt")
    locality_ground_truth = first_answer(record.get("locality_ground_truth"))
    if locality_prompt is not None and locality_ground_truth is not None:
        locality_groups["Relation_Specificity"] = {
            "prompt": [str(locality_prompt)],
            "ground_truth": [locality_ground_truth],
        }

    return locality_groups


def _build_locality_eval_items(locality_groups: dict[str, dict[str, list[str]]]) -> list[dict[str, Any]]:
    locality_items: list[dict[str, Any]] = []
    for locality_key, locality_group in locality_groups.items():
        prompts = locality_group.get("prompt", [])
        ground_truths = locality_group.get("ground_truth", [])
        for prompt, ground_truth in zip(prompts, ground_truths):
            locality_items.append(
                {
                    "key": locality_key,
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                }
            )
    return locality_items


def _get_first_locality_item(locality_items: list[dict[str, Any]]) -> tuple[str | None, Any]:
    if not locality_items:
        return None, None
    first_item = locality_items[0]
    return first_item.get("prompt"), first_item.get("ground_truth")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answer_matches(prediction: str | None, expected: Any) -> bool | None:
    answers = flatten_answers(expected)
    if not answers:
        return None
    pred_norm = normalize_text(prediction)
    if not pred_norm:
        return False
    for answer in answers:
        answer_norm = normalize_text(answer)
        if not answer_norm:
            continue
        if pred_norm == answer_norm:
            return True
        if pred_norm.startswith(answer_norm):
            return True
        if answer_norm in pred_norm:
            return True
    return False


def safe_mean(values: Iterable[bool | int | float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def model_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    return next(model.parameters()).device


def clear_memory(*objects: Any) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda:0",
    trust_remote_code: bool = True,
    dtype: str = "auto",
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype_arg: Any = dtype
    if dtype not in {"auto", None}:
        dtype_arg = getattr(torch, dtype)

    model_load_kwargs = {
        "trust_remote_code": trust_remote_code,
    }

    # Compatibility:
    # - newer transformers prefer `dtype=...`
    # - older transformers still expect `torch_dtype=...`
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype_arg,
            **model_load_kwargs,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'dtype'" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_arg,
            **model_load_kwargs,
        )

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested, but torch.cuda.is_available() is False. "
                "This Python environment currently has no usable CUDA backend. "
                "Use --device cpu for a quick workaround, or install a CUDA-enabled PyTorch build "
                "in the same environment before using --device cuda:0."
            )
        model = model.to(device)
    else:
        model = model.to(device)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 16,
    batch_size: int = 4,
    progress_desc: str | None = None,
) -> list[str]:
    if not prompts:
        return []

    outputs: list[str] = []
    device = model_device(model)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    batch_starts = range(0, len(prompts), batch_size)
    if progress_desc:
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        batch_starts = tqdm(
            batch_starts,
            total=total_batches,
            desc=progress_desc,
            unit="batch",
        )

    for start in batch_starts:
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = encoded.to(device)

        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        input_width = encoded["input_ids"].shape[1]
        for row_idx in range(len(batch_prompts)):
            generated_ids = generated[row_idx]
            continuation_ids = generated_ids[input_width:]

            text = tokenizer.decode(
                continuation_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            outputs.append(text.strip())

    tokenizer.padding_side = original_padding_side
    return outputs


def standardize_custom_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    standardized: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        locality_groups = _extract_locality_groups_from_record(record)
        locality_items = _build_locality_eval_items(locality_groups)
        locality_prompt, locality_ground_truth = _get_first_locality_item(locality_items)
        standardized.append(
            {
                "source_index": idx,
                "subject": record.get("subject"),
                "prompt": record["prompt"],
                "target_new": record["target_new"],
                "ground_truth": record.get("ground_truth"),
                "rephrase_prompt": record.get("rephrase_prompt"),
                "locality_prompt": locality_prompt,
                "locality_ground_truth": locality_ground_truth,
                "locality": locality_groups,
                "locality_eval": locality_items,
            }
        )
    return standardized


def _extract_rephrase(record: dict[str, Any]) -> str | None:
    if record.get("rephrase_prompt"):
        return record["rephrase_prompt"]
    rephrase = record.get("rephrase")
    if isinstance(rephrase, str):
        return rephrase
    if isinstance(rephrase, list) and rephrase:
        first_item = rephrase[0]
        if isinstance(first_item, str):
            return first_item
    return None


def standardize_external_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    standardized: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        locality_groups = _extract_locality_groups_from_record(record)
        locality_items = _build_locality_eval_items(locality_groups)
        locality_prompt, locality_ground_truth = _get_first_locality_item(locality_items)
        standardized.append(
            {
                "source_index": idx,
                "subject": record.get("subject"),
                "prompt": record["prompt"],
                "target_new": record["target_new"],
                "ground_truth": record.get("ground_truth"),
                "rephrase_prompt": _extract_rephrase(record),
                "locality_prompt": locality_prompt,
                "locality_ground_truth": locality_ground_truth,
                "locality": locality_groups,
                "locality_eval": locality_items,
            }
        )
    return standardized


def load_records_auto(path: str | Path) -> list[dict[str, Any]]:
    records = load_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"JSON must be a non-empty list: {path}")
    first = records[0]
    # Custom records explicitly carry flattened locality fields.
    # External KnowEdit/ZsRE-style records usually carry nested `locality`.
    if "locality_prompt" in first or "locality_ground_truth" in first:
        return standardize_custom_records(records)
    return standardize_external_records(records)


def filter_records_for_editing(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = []
    for record in records:
        if not record.get("prompt"):
            continue
        if first_answer(record.get("target_new")) is None:
            continue
        if record.get("rephrase_prompt") is None:
            continue
        if not record.get("locality_eval"):
            continue
        filtered.append(record)
    return filtered


def sample_records(records: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size >= len(records):
        return list(records)
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    chosen = sorted(indices[:sample_size])
    return [records[i] for i in chosen]


def build_locality_inputs(records: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]] | None:
    locality_keys: list[str] = []
    seen_keys: set[str] = set()
    for record in records:
        for locality_key, locality_group in record.get("locality", {}).items():
            if locality_key in seen_keys:
                continue
            if locality_group.get("prompt") and locality_group.get("ground_truth"):
                locality_keys.append(locality_key)
                seen_keys.add(locality_key)

    if not locality_keys:
        return None

    locality_inputs: dict[str, dict[str, list[Any]]] = {}
    for locality_key in locality_keys:
        prompts_per_record: list[Any] = []
        ground_truths_per_record: list[Any] = []
        has_any = False

        for record in records:
            locality_group = record.get("locality", {}).get(locality_key)
            if locality_group and locality_group.get("prompt") and locality_group.get("ground_truth"):
                prompts_per_record.append(list(locality_group["prompt"]))
                ground_truths_per_record.append(list(locality_group["ground_truth"]))
                has_any = True
            else:
                prompts_per_record.append(None)
                ground_truths_per_record.append(None)

        if has_any:
            locality_inputs[locality_key] = {
                "prompt": prompts_per_record,
                "ground_truth": ground_truths_per_record,
            }

    return locality_inputs or None


def evaluate_standardized_records(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_new_tokens: int = 16,
    batch_size: int = 4,
    progress_prefix: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    direct_prompts = [record["prompt"] for record in records]
    direct_outputs = generate_batch(
        model,
        tokenizer,
        direct_prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        progress_desc=None if not progress_prefix else f"{progress_prefix}: direct",
    )

    rephrase_indices = [idx for idx, record in enumerate(records) if record.get("rephrase_prompt")]
    rephrase_outputs_map: dict[int, str] = {}
    if rephrase_indices:
        rephrase_prompts = [records[idx]["rephrase_prompt"] for idx in rephrase_indices]
        rephrase_outputs = generate_batch(
            model,
            tokenizer,
            rephrase_prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            progress_desc=None if not progress_prefix else f"{progress_prefix}: rephrase",
        )
        rephrase_outputs_map = dict(zip(rephrase_indices, rephrase_outputs))

    locality_refs: list[tuple[int, dict[str, Any]]] = []
    locality_prompts: list[str] = []
    for idx, record in enumerate(records):
        for locality_item in record.get("locality_eval", []):
            locality_refs.append((idx, locality_item))
            locality_prompts.append(locality_item["prompt"])

    locality_results_map: dict[int, list[dict[str, Any]]] = {}
    if locality_refs:
        locality_outputs = generate_batch(
            model,
            tokenizer,
            locality_prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            progress_desc=None if not progress_prefix else f"{progress_prefix}: locality",
        )
        for (idx, locality_item), locality_output in zip(locality_refs, locality_outputs):
            locality_results_map.setdefault(idx, []).append(
                {
                    "key": locality_item["key"],
                    "prompt": locality_item["prompt"],
                    "ground_truth": locality_item["ground_truth"],
                    "output": locality_output,
                    "ground_truth_hit": answer_matches(locality_output, locality_item["ground_truth"]),
                }
            )

    evaluated_records: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        direct_output = direct_outputs[idx]
        rephrase_output = rephrase_outputs_map.get(idx)
        locality_results = locality_results_map.get(idx, [])
        first_locality_result = locality_results[0] if locality_results else None
        locality_mean_hit = safe_mean(item.get("ground_truth_hit") for item in locality_results)
        evaluated_records.append(
            {
                "source_index": record.get("source_index", idx),
                "subject": record.get("subject"),
                "prompt": record["prompt"],
                "target_new": flatten_answers(record.get("target_new")),
                "ground_truth": flatten_answers(record.get("ground_truth")),
                "rephrase_prompt": record.get("rephrase_prompt"),
                "locality_prompt": record.get("locality_prompt"),
                "locality_ground_truth": flatten_answers(record.get("locality_ground_truth")),
                "prompt_output": direct_output,
                "prompt_target_hit": answer_matches(direct_output, record.get("target_new")),
                "prompt_ground_truth_hit": answer_matches(direct_output, record.get("ground_truth")),
                "rephrase_output": rephrase_output,
                "rephrase_target_hit": answer_matches(rephrase_output, record.get("target_new")),
                "locality_output": None if first_locality_result is None else first_locality_result["output"],
                "locality_ground_truth_hit": locality_mean_hit,
                "locality_results": locality_results,
                "locality_n_prompts": len(locality_results),
            }
        )

    summary = summarize_eval_records(evaluated_records)
    return evaluated_records, summary


def summarize_eval_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_samples": len(records),
        "efficacy_es": safe_mean(record.get("prompt_target_hit") for record in records),
        "generalization_ps": safe_mean(record.get("rephrase_target_hit") for record in records),
        "locality_ns": safe_mean(record.get("locality_ground_truth_hit") for record in records),
        "old_fact_hit_rate": safe_mean(record.get("prompt_ground_truth_hit") for record in records),
    }


def summarize_stage_records(records: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    stage_records = [record[stage] for record in records if stage in record]
    return summarize_eval_records(stage_records)
