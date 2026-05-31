from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def clean_generated_text(text: str) -> str:
    marker = "Assistant:"
    cleaned = text.strip()
    if marker not in cleaned:
        return cleaned
    return cleaned.rsplit(marker, maxsplit=1)[-1].strip()


def build_user_message(*, image: Any, prompt: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }


def build_inference_messages(*, image: Any, prompt: str) -> list[dict[str, Any]]:
    return [build_user_message(image=image, prompt=prompt)]


def build_training_messages(
    sample: Mapping[str, Any],
    prompt: str,
) -> list[dict[str, Any]]:
    return [
        build_user_message(image=sample["image"], prompt=prompt),
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": str(sample["caption"]).strip()},
            ],
        },
    ]


def align_model_generation_config_with_tokenizer(model: Any, tokenizer: Any) -> None:
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, name, None)
        if token_id is None:
            continue
        if hasattr(model, "config"):
            setattr(model.config, name, token_id)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            setattr(generation_config, name, token_id)


def resolve_auto_model_class(transformers_module: Any) -> Any:
    auto_model_class = getattr(transformers_module, "AutoModelForImageTextToText", None)
    if auto_model_class is not None:
        return auto_model_class
    return transformers_module.AutoModelForVision2Seq


def resolve_torch_dtype(*, device: Any, torch_dtype: Any | None) -> Any | None:
    if torch_dtype != "auto":
        return torch_dtype
    if getattr(device, "type", None) != "cuda":
        return None

    import torch

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_pretrained_model(
    auto_model_class: Any,
    model_id: str,
    *,
    torch_dtype: Any | None,
    attn_implementation: str | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if attn_implementation is not None:
        kwargs["_attn_implementation"] = attn_implementation
    if torch_dtype is None:
        return auto_model_class.from_pretrained(model_id, **kwargs)
    try:
        return auto_model_class.from_pretrained(model_id, dtype=torch_dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return auto_model_class.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)


def load_model_and_processor(
    *,
    model_id: str,
    device: Any,
    torch_dtype: Any | None = None,
    prefer_flash_attention: bool = False,
    transformers_module: Any | None = None,
) -> dict[str, Any]:
    if transformers_module is None:
        import transformers as transformers_module

    torch_dtype = resolve_torch_dtype(device=device, torch_dtype=torch_dtype)
    processor = transformers_module.AutoProcessor.from_pretrained(model_id)
    auto_model_class = resolve_auto_model_class(transformers_module)

    attn_candidates = ["eager"]
    if getattr(device, "type", None) == "cuda" and prefer_flash_attention:
        attn_candidates.insert(0, "flash_attention_2")

    last_error: Exception | None = None
    for attn_implementation in attn_candidates:
        try:
            model = load_pretrained_model(
                auto_model_class,
                model_id,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
            )
            model = model.to(device)
            align_model_generation_config_with_tokenizer(model, processor.tokenizer)
            return {
                "processor": processor,
                "model": model,
                "torch_dtype": torch_dtype,
                "attn_implementation": attn_implementation,
            }
        except Exception as exc:
            last_error = exc

    assert last_error is not None
    raise RuntimeError(f"Failed to load {model_id}") from last_error


def enable_input_grads_for_gradient_checkpointing(model: Any) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_input_embeddings):
        return
    input_embeddings = get_input_embeddings()
    if input_embeddings is None:
        return

    def _make_inputs_require_grad(_module: Any, _inputs: Any, output: Any) -> None:
        if hasattr(output, "requires_grad_"):
            output.requires_grad_(True)

    input_embeddings.register_forward_hook(_make_inputs_require_grad)


def prepare_model_for_gradient_checkpointing_training(
    model: Any,
    *,
    use_gradient_checkpointing: bool,
) -> None:
    if use_gradient_checkpointing:
        enable_input_grads_for_gradient_checkpointing(model)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False


def ensure_rgb(image: Any) -> Any:
    if hasattr(image, "mode") and image.mode != "RGB":
        return image.convert("RGB")
    return image


def build_sft_collate_fn(
    processor: Any,
    *,
    image_token_id: int | None,
    prompt: str,
) -> Any:
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [
            processor.apply_chat_template(
                build_training_messages(sample=sample, prompt=prompt),
                tokenize=False,
                add_generation_prompt=False,
            )
            for sample in batch
        ]
        prompt_texts = [
            processor.apply_chat_template(
                build_inference_messages(image=sample["image"], prompt=prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for sample in batch
        ]
        images = [[ensure_rgb(sample["image"])] for sample in batch]
        tokenizer = processor.tokenizer
        original_padding_side = getattr(tokenizer, "padding_side", None)
        if original_padding_side is not None:
            tokenizer.padding_side = "right"
        try:
            encoded = processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            prompt_encoded = processor(
                text=prompt_texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        finally:
            if original_padding_side is not None:
                tokenizer.padding_side = original_padding_side

        labels = encoded["input_ids"].new_full(encoded["input_ids"].shape, -100)
        sequence_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        prompt_lengths = prompt_encoded["attention_mask"].sum(dim=1).tolist()
        lengths = zip(prompt_lengths, sequence_lengths, strict=True)
        for row_index, (prompt_length, sequence_length) in enumerate(lengths):
            labels[row_index, int(prompt_length) : int(sequence_length)] = encoded["input_ids"][
                row_index,
                int(prompt_length) : int(sequence_length),
            ]
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        encoded["labels"] = labels
        return encoded

    return collate_fn


def move_batch_to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _coerce_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _prompt_lengths(encoded: Mapping[str, Any]) -> list[int]:
    input_ids = encoded["input_ids"]
    batch_size = _coerce_int(input_ids.shape[0])
    width = _coerce_int(input_ids.shape[1])
    return [width] * batch_size


def generate_captions_for_batch(
    *,
    model: Any,
    processor: Any,
    images: Sequence[Any],
    device: Any,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool = False,
) -> list[str]:
    if not images:
        return []

    import torch

    prompt_texts = [
        processor.apply_chat_template(
            build_inference_messages(image=image, prompt=prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
        for image in images
    ]
    tokenizer = getattr(processor, "tokenizer", None)
    original_padding_side = getattr(tokenizer, "padding_side", None)
    if original_padding_side is not None:
        tokenizer.padding_side = "left"
    try:
        encoded = processor(
            text=prompt_texts,
            images=[[ensure_rgb(image)] for image in images],
            padding=True,
            return_tensors="pt",
        )
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
    encoded = move_batch_to_device(encoded, device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    sequences = [
        generated_ids[row_index, prompt_length:]
        for row_index, prompt_length in enumerate(_prompt_lengths(encoded))
    ]
    decoded = processor.batch_decode(sequences, skip_special_tokens=True)
    return [clean_generated_text(text) for text in decoded]


def generate_captions(
    *,
    model: Any,
    processor: Any,
    samples: Sequence[Mapping[str, Any]],
    device: Any,
    prompt: str,
    max_new_tokens: int,
    batch_size: int = 1,
    progress_desc: str | None = None,
    logger: Any | None = None,
) -> list[str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    from .logger_utils import progress_iterable

    predictions: list[str] = []
    sample_batches = [
        samples[start_index : start_index + batch_size]
        for start_index in range(0, len(samples), batch_size)
    ]
    for sample_batch in progress_iterable(sample_batches, desc=progress_desc, logger=logger):
        predictions.extend(
            generate_captions_for_batch(
                model=model,
                processor=processor,
                images=[sample["image"] for sample in sample_batch],
                device=device,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        )
    return predictions


def generate_caption_splits(
    *,
    label: str,
    model: Any,
    processor: Any,
    sample_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    device: Any,
    prompt: str,
    max_new_tokens: int,
    batch_size: int,
    logger: Any | None = None,
) -> dict[str, list[str]]:
    model.eval()
    outputs: dict[str, list[str]] = {}
    for output_key, samples in sample_splits.items():
        if logger is not None:
            logger.info("Running %s caption generation for %s samples", label.lower(), output_key)
        outputs[output_key] = generate_captions(
            model=model,
            processor=processor,
            samples=samples,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            progress_desc=f"{label} {output_key} captions",
            logger=logger,
        )
    return outputs


def save_caption_comparison_figure(
    *,
    samples: list[dict[str, Any]],
    zero_shot_predictions: list[str],
    fine_tuned_predictions: list[str],
    output_path: Any,
) -> None:
    if not samples:
        return

    from .visualization import require_matplotlib, show_caption_comparison_cards

    plt = require_matplotlib()
    figure, _ = show_caption_comparison_cards(
        samples,
        prediction_sets={
            "Zero-shot": zero_shot_predictions,
            "Fine-tuned": fine_tuned_predictions,
        },
        show=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


__all__ = [
    "align_model_generation_config_with_tokenizer",
    "build_inference_messages",
    "build_sft_collate_fn",
    "build_training_messages",
    "build_user_message",
    "clean_generated_text",
    "enable_input_grads_for_gradient_checkpointing",
    "ensure_rgb",
    "generate_caption_splits",
    "generate_captions",
    "generate_captions_for_batch",
    "load_model_and_processor",
    "load_pretrained_model",
    "move_batch_to_device",
    "prepare_model_for_gradient_checkpointing_training",
    "resolve_auto_model_class",
    "resolve_torch_dtype",
    "save_caption_comparison_figure",
]
