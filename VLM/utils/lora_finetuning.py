from __future__ import annotations

from typing import Any

import peft

from .caption_metrics import (
    build_caption_before_after_report,
    build_comparison_metrics,
    safe_compute_caption_metric_comparison,
)
from .device_utils import release_cuda_memory
from .report_utils import log_before_after_report, log_parameter_efficiency
from .smolvlm_utils import (
    align_model_generation_config_with_tokenizer,
    generate_caption_splits,
    load_pretrained_model,
    save_caption_comparison_figure,
)
from .training_utils import build_rank_summary_row, build_sft_trainer_state, count_parameters, save_json


def train_and_evaluate_lora_rank(
    *,
    rank: int,
    config: Any,
    processor: Any,
    auto_model_class: Any,
    transformers_module: Any,
    torch_module: Any,
    torch_dtype: Any | None,
    device: Any,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    comparison_samples: list[dict[str, Any]],
    metric_samples: list[dict[str, Any]],
    zero_shot_comparison_predictions: list[str],
    zero_shot_metric_predictions: list[str],
    full_finetuning_trainable_params: int | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    rank = int(rank)
    rank_output_dir = config.output_dir / f"rank-{rank}"
    rank_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        base_model = load_pretrained_model(auto_model_class, config.model_name, torch_dtype=torch_dtype)
        align_model_generation_config_with_tokenizer(base_model, processor.tokenizer)
        if full_finetuning_trainable_params is None:
            full_finetuning_trainable_params = count_parameters(base_model)

        lora_config = peft.LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=list(config.target_modules),
            lora_dropout=0.05,
            bias="none",
        )
        lora_model = peft.get_peft_model(base_model, lora_config)
        lora_model.to(device)
        align_model_generation_config_with_tokenizer(lora_model, processor.tokenizer)

        trainer_state = build_sft_trainer_state(
            transformers_module=transformers_module,
            config=config,
            model=lora_model,
            processor=processor,
            device=device,
            torch_dtype=torch_dtype,
            train_rows=train_rows,
            val_rows=val_rows,
            output_dir=rank_output_dir,
        )
        if logger is not None:
            logger.info(
                "Starting LoRA fine-tuning | rank=%s train_rows=%s val_rows=%s effective_batch=%s estimated_optimizer_steps=%s",
                rank,
                len(train_rows),
                len(val_rows),
                trainer_state["effective_batch_size"],
                trainer_state["estimated_steps"],
            )
        trainer = trainer_state["trainer"]
        train_result = trainer.train()
        if logger is not None:
            logger.info("Training complete, starting evaluation | rank=%s", rank)
        eval_metrics = trainer.evaluate()
        trainer.save_model()

        trainable_params = count_parameters(lora_model, trainable_only=True)
        total_params = count_parameters(lora_model)
        trainable_ratio = float(trainable_params / total_params) if total_params else 0.0
        if logger is not None:
            logger.info(
                "LoRA trainable parameters | trainable_params=%s all_params=%s trainable_ratio=%.4f%%",
                int(trainable_params),
                int(total_params),
                100 * trainable_ratio,
            )

        fine_tuned_outputs = generate_caption_splits(
            label=f"LoRA rank {rank}",
            model=lora_model,
            processor=processor,
            sample_splits={
                "comparison": comparison_samples,
                "metric": metric_samples,
            },
            device=device,
            prompt=config.prompt,
            max_new_tokens=config.max_new_tokens,
            batch_size=config.per_device_eval_batch_size,
            logger=logger,
        )
        caption_scores = safe_compute_caption_metric_comparison(
            zero_shot_predictions=zero_shot_metric_predictions,
            fine_tuned_predictions=fine_tuned_outputs["metric"],
            samples=metric_samples,
            logger=logger,
        )
        before_after_report = build_caption_before_after_report(
            samples=comparison_samples,
            zero_shot_predictions=zero_shot_comparison_predictions,
            fine_tuned_predictions=fine_tuned_outputs["comparison"],
            metrics=build_comparison_metrics(caption_scores),
        )
        save_json(rank_output_dir / "before_after.json", before_after_report)
        save_caption_comparison_figure(
            samples=comparison_samples,
            zero_shot_predictions=zero_shot_comparison_predictions,
            fine_tuned_predictions=fine_tuned_outputs["comparison"],
            output_path=rank_output_dir / "before_after.png",
        )
        save_json(
            rank_output_dir / "train_eval_metrics.json",
            {
                "rank": rank,
                "train_result": dict(train_result.metrics),
                "eval_metrics": eval_metrics,
                "log_history": list(trainer.state.log_history),
            },
        )
        save_json(rank_output_dir / "caption_scores.json", caption_scores)
        if logger is not None:
            log_before_after_report(before_after_report, logger)

        summary = build_rank_summary_row(
            rank=rank,
            trainable_params=trainable_params,
            total_params=total_params,
            full_finetuning_trainable_params=int(full_finetuning_trainable_params or 0),
            eval_metrics=eval_metrics,
            caption_scores=caption_scores,
        )
        if logger is not None:
            log_parameter_efficiency(summary, logger)
        return {
            "summary": summary,
            "before_after_report": before_after_report,
            "caption_scores": caption_scores,
            "eval_metrics": eval_metrics,
            "full_finetuning_trainable_params": int(full_finetuning_trainable_params or 0),
            "rank_output_dir": str(rank_output_dir),
        }
    finally:
        try:
            del trainer
        except UnboundLocalError:
            pass
        try:
            del trainer_state
        except UnboundLocalError:
            pass
        try:
            del lora_model
        except UnboundLocalError:
            pass
        try:
            del base_model
        except UnboundLocalError:
            pass
        release_cuda_memory(device, torch_module)


__all__ = ["train_and_evaluate_lora_rank"]
