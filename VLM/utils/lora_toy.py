from __future__ import annotations

from typing import Any


def require_lora_value(name: str, value: Any, hint: str) -> Any:
    if value is None:
        raise ValueError(f"[LoRA 빈칸] {name}가 None입니다. {hint}")
    return value


def _require_shape(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if actual != expected:
        raise ValueError(f"[LoRA shape] {name} shape가 {actual}입니다. 기대값은 {expected}입니다.")


def validate_lora_layer(
    layer: Any,
    base_layer: Any,
    rank: int,
    probe: Any,
    *,
    logger: Any | None = None,
) -> None:
    import torch

    expected_lora_a_shape = (rank, base_layer.in_features)
    expected_lora_b_shape = (base_layer.out_features, rank)
    _require_shape("lora_a.weight", tuple(layer.lora_a.weight.shape), expected_lora_a_shape)
    _require_shape("lora_b.weight", tuple(layer.lora_b.weight.shape), expected_lora_b_shape)

    trainable_names = {name for name, parameter in layer.named_parameters() if parameter.requires_grad}
    expected_trainable_names = {"lora_a.weight", "lora_b.weight"}
    if trainable_names != expected_trainable_names:
        raise ValueError(
            "[LoRA freeze] trainable parameter가 "
            f"{sorted(trainable_names)}입니다. 기대값은 {sorted(expected_trainable_names)}입니다."
        )

    expected_scaling = float(layer.alpha) / float(rank)
    if abs(float(layer.scaling) - expected_scaling) > 1e-12:
        raise ValueError(
            "[LoRA scaling] scaling 값이 "
            f"{float(layer.scaling):.6f}입니다. 기대값은 alpha/rank={expected_scaling:.6f}입니다."
        )

    try:
        base_output = base_layer(probe)
        lora_output = layer(probe)
    except Exception as error:
        raise ValueError(
            "[LoRA forward] 출력 shape 또는 초기 base 출력 동등성 검증에 실패했습니다. "
            "adapter update와 scaling 적용 순서를 확인하세요."
        ) from error

    if tuple(lora_output.shape) != tuple(base_output.shape):
        raise ValueError(
            "[LoRA forward] 출력 shape가 "
            f"{tuple(lora_output.shape)}입니다. 기대값은 {tuple(base_output.shape)}입니다."
        )

    max_abs_diff = (base_output - lora_output).abs().max().item()
    if max_abs_diff > 1e-6:
        raise ValueError(
            "[LoRA forward] 초기 LoRA 출력이 base 출력과 다릅니다. "
            f"max_abs_diff={max_abs_diff:.6e}. lora_b 초기화와 forward 식을 확인하세요."
        )

    original_lora_b_weight = layer.lora_b.weight.detach().clone()
    try:
        with torch.no_grad():
            layer.lora_b.weight.fill_(0.1)
        expected_output = base_layer(probe) + expected_scaling * layer.lora_b(layer.lora_a(probe))
        actual_output = layer(probe)
    except Exception as error:
        raise ValueError(
            "[LoRA forward] 출력 shape 또는 초기 base 출력 동등성 검증에 실패했습니다. "
            "adapter update와 scaling 적용 순서를 확인하세요."
        ) from error
    finally:
        with torch.no_grad():
            layer.lora_b.weight.copy_(original_lora_b_weight)

    update_max_abs_diff = (expected_output - actual_output).abs().max().item()
    if update_max_abs_diff > 1e-6:
        raise ValueError(
            "[LoRA forward] non-zero adapter update가 기대 식과 다릅니다. "
            f"max_abs_diff={update_max_abs_diff:.6e}. base 출력, LoRA update, scaling 조합을 확인하세요."
        )

    if logger is not None:
        logger.info("lora_a.weight shape: %s", tuple(layer.lora_a.weight.shape))
        logger.info("lora_b.weight shape: %s", tuple(layer.lora_b.weight.shape))
        logger.info("scaling: %.6f", float(layer.scaling))
        logger.info("trainable parameters: %s", sorted(trainable_names))
        logger.info("initial base-vs-LoRA max_abs_diff: %.6e", max_abs_diff)
        logger.info("non-zero update validation max_abs_diff: %.6e", update_max_abs_diff)


def _smooth_basis(length: int, components: int, device: Any) -> Any:
    import torch

    positions = torch.linspace(0.0, 1.0, steps=length, device=device)
    waves = []
    for index in range(components):
        frequency = (index // 2) + 1
        if index % 2 == 0:
            wave = torch.sin(torch.pi * frequency * positions)
        else:
            wave = torch.cos(torch.pi * frequency * positions)
        waves.append(wave)
    raw_basis = torch.stack(waves, dim=1)
    basis, _ = torch.linalg.qr(raw_basis, mode="reduced")
    return basis[:, :components]


def _structured_matrix(
    *,
    rows: int,
    columns: int,
    rank: int,
    min_singular_value: float,
    max_singular_value: float,
    device: Any,
) -> Any:
    import torch

    effective_rank = min(rank, rows, columns)
    row_basis = _smooth_basis(rows, effective_rank, device)
    column_basis = _smooth_basis(columns, effective_rank, device)
    singular_values = torch.linspace(
        max_singular_value,
        min_singular_value,
        steps=effective_rank,
        device=device,
    )
    return (row_basis * singular_values.unsqueeze(0)) @ column_basis.T


def _structured_samples(
    *,
    samples: int,
    features: int,
    device: Any,
    generator: Any | None,
    noise_scale: float,
) -> Any:
    import torch

    sample_positions = torch.linspace(0.0, 1.0, steps=samples, device=device).unsqueeze(1)
    feature_positions = torch.linspace(0.0, 1.0, steps=features, device=device).unsqueeze(0)
    pattern = (
        torch.sin(2.0 * torch.pi * (1.0 * sample_positions + 0.35 * feature_positions))
        + 0.6 * torch.cos(2.0 * torch.pi * (0.5 * sample_positions - 0.75 * feature_positions))
        + 0.35 * torch.sin(2.0 * torch.pi * (2.0 * sample_positions + 0.12 * feature_positions))
    )
    noise = noise_scale * torch.randn(samples, features, device=device, generator=generator)
    return pattern + noise


def make_problem(
    config: Any,
    device: Any,
    *,
    generator: Any | None = None,
) -> dict[str, Any]:
    import torch

    base_weight = _structured_matrix(
        rows=config.output_dim,
        columns=config.input_dim,
        rank=min(4, config.output_dim, config.input_dim),
        min_singular_value=0.25,
        max_singular_value=1.00,
        device=device,
    )
    low_rank_update = _structured_matrix(
        rows=config.output_dim,
        columns=config.input_dim,
        rank=config.true_rank,
        min_singular_value=0.35,
        max_singular_value=1.40,
        device=device,
    )
    residual_scale = float(getattr(config, "residual_scale", 0.0))
    dense_residual = residual_scale * torch.randn(
        config.output_dim,
        config.input_dim,
        device=device,
        generator=generator,
    )
    target_weight = base_weight + low_rank_update + dense_residual
    x_train = _structured_samples(
        samples=config.train_samples,
        features=config.input_dim,
        device=device,
        generator=generator,
        noise_scale=0.04,
    )
    x_eval = _structured_samples(
        samples=config.eval_samples,
        features=config.input_dim,
        device=device,
        generator=generator,
        noise_scale=0.04,
    )
    return {
        "base_weight": base_weight,
        "target_weight": target_weight,
        "x_train": x_train,
        "y_train": x_train @ target_weight.T,
        "x_eval": x_eval,
        "y_eval": x_eval @ target_weight.T,
    }


__all__ = ["make_problem", "require_lora_value", "validate_lora_layer"]
