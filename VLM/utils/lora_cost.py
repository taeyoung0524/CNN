from __future__ import annotations

from typing import Any, Sequence

import torch

FP16_WEIGHT_BYTES = 2
FP32_MASTER_WEIGHT_BYTES = 4
FP32_GRADIENT_BYTES = 4
ADAM_STATE_BYTES = 8
BYTE_UNITS = ("B", "KB", "MB", "GB", "TB")


def format_bytes(num_bytes: float | int) -> str:
    value = float(num_bytes)
    unit_index = 0
    while abs(value) >= 1024.0 and unit_index < len(BYTE_UNITS) - 1:
        value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {BYTE_UNITS[unit_index]}"
    return f"{value:.2f} {BYTE_UNITS[unit_index]}"


def load_model_from_config(model_id: str) -> Any:
    from transformers import AutoConfig
    import transformers

    from .smolvlm_utils import resolve_auto_model_class

    config = AutoConfig.from_pretrained(model_id)
    auto_model_class = resolve_auto_model_class(transformers)
    return auto_model_class.from_config(config)


def build_parameter_report(*, total_params: int, trainable_params: int) -> dict[str, float | int]:
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_ratio": float(trainable_params / total_params) if total_params else 0.0,
    }


def build_lora_report(
    *,
    total_params: int,
    trainable_params: int,
    rank: int,
    target_module_count: int,
) -> dict[str, float | int]:
    report = build_parameter_report(total_params=total_params, trainable_params=trainable_params)
    report["rank"] = int(rank)
    report["target_module_count"] = int(target_module_count)
    return report


def collect_target_module_names(
    model: Any,
    *,
    prefix: str,
    suffixes: Sequence[str],
) -> tuple[str, ...]:
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith(prefix)
        and name.endswith(tuple(suffixes))
    ]
    return tuple(names)


def collect_linear_shapes(model: Any, module_names: Sequence[str]) -> tuple[tuple[int, int], ...]:
    modules = dict(model.named_modules())
    return tuple(
        (int(modules[module_name].in_features), int(modules[module_name].out_features))
        for module_name in module_names
    )


def estimate_lora_trainable_params(
    *,
    linear_shapes: Sequence[tuple[int, int]],
    rank: int,
) -> int:
    return sum(int(rank) * (in_features + out_features) for in_features, out_features in linear_shapes)


def build_rank_cost_records(
    *,
    linear_shapes: Sequence[tuple[int, int]],
    ranks: Sequence[int],
    full_total_params: int,
) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    for rank in ranks:
        trainable_params = estimate_lora_trainable_params(
            linear_shapes=linear_shapes,
            rank=int(rank),
        )
        total_params = int(full_total_params) + int(trainable_params)
        records.append(
            {
                "rank": int(rank),
                "trainable_params": int(trainable_params),
                "total_params": total_params,
                "trainable_ratio": float(trainable_params / total_params) if total_params else 0.0,
                "vs_full_finetuning_ratio": (
                    float(trainable_params / full_total_params) if full_total_params else 0.0
                ),
            }
        )
    return records


def memory_profile_assumptions() -> dict[str, str | int]:
    return {
        "scope": "parameter_state_only",
        "included": (
            "fp16 model weights for frozen and trainable params, plus fp32 master weights, "
            "gradients, and Adam optimizer states for trainable params"
        ),
        "excluded": "activations, dataloader memory, KV cache, sequence length effects, fragmentation",
        "frozen_weight_bytes": FP16_WEIGHT_BYTES,
        "trainable_weight_bytes": FP16_WEIGHT_BYTES + FP32_MASTER_WEIGHT_BYTES,
        "gradient_bytes": FP32_GRADIENT_BYTES,
        "adam_state_bytes": ADAM_STATE_BYTES,
    }


def estimate_memory_profiles(*, total_params: int, trainable_params: int) -> dict[str, dict[str, int]]:
    frozen_params = int(total_params) - int(trainable_params)
    full_weight_bytes = (FP16_WEIGHT_BYTES + FP32_MASTER_WEIGHT_BYTES) * int(total_params)
    full_gradient_bytes = FP32_GRADIENT_BYTES * int(total_params)
    full_optimizer_bytes = ADAM_STATE_BYTES * int(total_params)
    lora_weight_bytes = (FP16_WEIGHT_BYTES * frozen_params) + (
        (FP16_WEIGHT_BYTES + FP32_MASTER_WEIGHT_BYTES) * int(trainable_params)
    )
    lora_gradient_bytes = FP32_GRADIENT_BYTES * int(trainable_params)
    lora_optimizer_bytes = ADAM_STATE_BYTES * int(trainable_params)
    return {
        "full_finetuning": {
            "weight_bytes": full_weight_bytes,
            "gradient_bytes": full_gradient_bytes,
            "optimizer_state_bytes": full_optimizer_bytes,
            "total_bytes": full_weight_bytes + full_gradient_bytes + full_optimizer_bytes,
        },
        "lora": {
            "weight_bytes": lora_weight_bytes,
            "gradient_bytes": lora_gradient_bytes,
            "optimizer_state_bytes": lora_optimizer_bytes,
            "total_bytes": lora_weight_bytes + lora_gradient_bytes + lora_optimizer_bytes,
        },
    }


__all__ = [
    "build_lora_report",
    "build_parameter_report",
    "build_rank_cost_records",
    "collect_linear_shapes",
    "collect_target_module_names",
    "estimate_lora_trainable_params",
    "estimate_memory_profiles",
    "format_bytes",
    "load_model_from_config",
    "memory_profile_assumptions",
]
