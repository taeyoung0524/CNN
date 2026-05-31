from __future__ import annotations

from dataclasses import asdict, is_dataclass
import inspect
import json
from pathlib import Path
from typing import Any


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def count_parameters(model: Any, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def build_training_arguments_kwargs(
    *,
    training_arguments_cls: Any,
    output_dir: str,
    config: Any,
    device: Any,
    torch_dtype: Any | None,
) -> dict[str, Any]:
    dtype_name = "" if torch_dtype is None else str(torch_dtype).lower()
    use_bf16 = device.type == "cuda" and "bfloat16" in dtype_name
    use_fp16 = device.type == "cuda" and not use_bf16 and "float16" in dtype_name
    kwargs = {
        "output_dir": output_dir,
        "remove_unused_columns": False,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_strategy": "epoch",
        "report_to": "none",
        "gradient_checkpointing": device.type == "cuda",
        "fp16": use_fp16,
        "bf16": use_bf16,
    }
    seed = getattr(config, "seed", None)
    if seed is not None:
        kwargs["seed"] = seed
        kwargs["data_seed"] = seed
    parameter_names = inspect.signature(training_arguments_cls.__init__).parameters
    if "disable_tqdm" in parameter_names:
        kwargs["disable_tqdm"] = False
    if "eval_strategy" in parameter_names:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return kwargs


def build_config_payload(config: Any, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_config = asdict(config) if is_dataclass(config) else dict(config)
    payload = {
        key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
        for key, value in raw_config.items()
    }
    if extra:
        payload.update(extra)
    return payload


def estimate_training_steps(
    *,
    train_rows: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
) -> dict[str, int]:
    effective_batch_size = int(per_device_train_batch_size) * int(gradient_accumulation_steps)
    estimated_steps = ((int(train_rows) + effective_batch_size - 1) // effective_batch_size) * int(num_train_epochs)
    return {
        "effective_batch_size": effective_batch_size,
        "estimated_steps": estimated_steps,
    }


def build_rank_summary_row(
    *,
    rank: int,
    trainable_params: int,
    total_params: int,
    full_finetuning_trainable_params: int,
    eval_metrics: dict[str, Any],
    caption_scores: dict[str, Any],
) -> dict[str, Any]:
    vs_full_ratio = (
        float(trainable_params / full_finetuning_trainable_params)
        if full_finetuning_trainable_params
        else 0.0
    )
    saved_params = int(full_finetuning_trainable_params - trainable_params)
    saved_ratio = (
        float(saved_params / full_finetuning_trainable_params)
        if full_finetuning_trainable_params
        else 0.0
    )
    reduction_factor = float(full_finetuning_trainable_params / trainable_params) if trainable_params else 0.0
    return {
        "rank": int(rank),
        "lora_alpha": int(rank) * 2,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "full_finetuning_trainable_params": int(full_finetuning_trainable_params),
        "trainable_ratio": float(trainable_params / total_params) if total_params else 0.0,
        "vs_full_finetuning_ratio": vs_full_ratio,
        "vs_full_finetuning_reduction_factor": reduction_factor,
        "vs_full_finetuning_saved_params": saved_params,
        "vs_full_finetuning_saved_ratio": saved_ratio,
        "eval_loss": float(eval_metrics["eval_loss"]),
        "caption_scores": caption_scores,
    }


def build_sft_trainer_state(
    *,
    transformers_module: Any,
    config: Any,
    model: Any,
    processor: Any,
    device: Any,
    torch_dtype: Any | None,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    from .coco_dataloader import create_coco_dataset
    from .smolvlm_utils import build_sft_collate_fn, prepare_model_for_gradient_checkpointing_training

    training_args = transformers_module.TrainingArguments(
        **build_training_arguments_kwargs(
            training_arguments_cls=transformers_module.TrainingArguments,
            output_dir=str(output_dir),
            config=config,
            device=device,
            torch_dtype=torch_dtype,
        )
    )
    prepare_model_for_gradient_checkpointing_training(
        model,
        use_gradient_checkpointing=bool(getattr(training_args, "gradient_checkpointing", False)),
    )
    collate_fn = build_sft_collate_fn(
        processor,
        image_token_id=getattr(model.config, "image_token_id", None),
        prompt=config.prompt,
    )
    trainer = transformers_module.Trainer(
        model=model,
        args=training_args,
        train_dataset=create_coco_dataset(train_rows),
        eval_dataset=create_coco_dataset(val_rows),
        data_collator=collate_fn,
        processing_class=processor,
    )
    return {
        "trainer": trainer,
        **estimate_training_steps(
            train_rows=len(train_rows),
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.num_train_epochs,
        ),
    }


__all__ = [
    "build_config_payload",
    "build_rank_summary_row",
    "build_sft_trainer_state",
    "build_training_arguments_kwargs",
    "count_parameters",
    "estimate_training_steps",
    "save_json",
]
