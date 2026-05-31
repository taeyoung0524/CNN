from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_fixed_width_table(
    *,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> str:
    string_rows = [
        {column: _format_cell(row.get(column)) for column in columns}
        for row in rows
    ]
    widths = {
        column: max([len(column), *(len(row[column]) for row in string_rows)])
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(row[column].ljust(widths[column]) for column in columns)
        for row in string_rows
    ]
    return "\n".join([header, separator, *body]) if body else "\n".join([header, separator])


def log_before_after_report(report: dict[str, Any], logger: Any) -> None:
    logger.info("Terminal before/after comparison")
    metrics = report.get("metrics", {})
    if "delta" in metrics:
        logger.info(
            "Metrics | bleu=%.4f->%.4f (delta=%.4f) meteor=%.4f->%.4f (delta=%.4f) cider_d=%.4f->%.4f (delta=%.4f)",
            metrics["zero_shot"]["bleu"],
            metrics["fine_tuned"]["bleu"],
            metrics["delta"]["bleu"],
            metrics["zero_shot"]["meteor"],
            metrics["fine_tuned"]["meteor"],
            metrics["delta"]["meteor"],
            metrics["zero_shot"]["cider_d"],
            metrics["fine_tuned"]["cider_d"],
            metrics["delta"]["cider_d"],
        )
    else:
        logger.info("Metrics | skipped=%s", metrics.get("skipped_reason", "caption metrics unavailable"))
    for index, record in enumerate(report.get("samples", []), start=1):
        logger.info("Sample %s | filename=%s", index, record.get("filename", "n/a"))
        logger.info("GT: %s", record.get("ground_truth", "n/a"))
        logger.info("Zero-shot: %s", record.get("zero_shot", "n/a"))
        logger.info("Fine-tuned: %s", record.get("fine_tuned", "n/a"))


def log_rank_summary(rows: list[dict[str, Any]], logger: Any) -> None:
    if not rows:
        return

    summary_rows = [
        {
            "rank": row["rank"],
            "trainable_params": row["trainable_params"],
            "trainable_ratio": f"{float(row['trainable_ratio']):.2%}",
            "eval_loss": f"{float(row['eval_loss']):.6f}",
        }
        for row in rows
    ]
    logger.info(
        "LoRA rank summary\n%s",
        build_fixed_width_table(
            rows=summary_rows,
            columns=("rank", "trainable_params", "trainable_ratio", "eval_loss"),
        ),
    )


def log_parameter_efficiency(summary: dict[str, Any], logger: Any) -> None:
    logger.info(
        "LoRA parameter efficiency | rank=%s lora_trainable_params=%s full_finetuning_trainable_params=%s "
        "vs_full_ratio=%.4f%% reduction_factor=%.2fx saved_params=%s saved_ratio=%.4f%%",
        summary["rank"],
        int(summary["trainable_params"]),
        int(summary["full_finetuning_trainable_params"]),
        100 * float(summary["vs_full_finetuning_ratio"]),
        float(summary["vs_full_finetuning_reduction_factor"]),
        int(summary["vs_full_finetuning_saved_params"]),
        100 * float(summary["vs_full_finetuning_saved_ratio"]),
    )


def load_full_finetuning_comparison(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "available": False,
            "path": str(path),
            "reason": "Run 1.5-VLM.ipynb to create full fine-tuning comparison artifacts.",
        }
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError) as exc:
        return {
            "available": False,
            "path": str(path),
            "reason": f"Could not read full fine-tuning comparison artifact: {exc}",
        }
    metrics = dict(report.get("metrics", {}))
    model_report = dict(report.get("models", {})).get("pytorch") or {}
    trainable_params = model_report.get("trainable_params") if isinstance(model_report, dict) else None
    return {
        "available": True,
        "path": str(path),
        "metrics": {
            "zero_shot": metrics.get("zero_shot"),
            "full_finetuned": metrics.get("pytorch_fine_tuned"),
            "full_ft_delta": metrics.get("pytorch_delta"),
        },
        "eval_loss": dict(report.get("eval_loss", {})).get("pytorch"),
        "trainable_params": trainable_params,
        "model": model_report,
        "samples": list(report.get("samples", [])),
    }


def _sample_identity(sample: dict[str, Any]) -> Any:
    for key in ("cocoid", "id", "filename"):
        if key in sample:
            return sample[key]
    return None


def _caption_sample_row(sample: dict[str, Any], zero_shot_prediction: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ground_truth": str(sample["caption"]),
        "zero_shot": str(zero_shot_prediction),
    }
    for key in ("cocoid", "filename"):
        if key in sample:
            row[key] = int(sample[key]) if key == "cocoid" else str(sample[key])
    return row


def _add_comparison_predictions(
    *,
    report_samples: list[dict[str, Any]],
    outputs: dict[str, list[str]] | None,
    field_name: str,
) -> None:
    if outputs is None:
        return
    comparison_predictions = outputs.get("comparison", [])
    for index, sample_report in enumerate(report_samples):
        if index < len(comparison_predictions):
            sample_report[field_name] = comparison_predictions[index]


def build_full_finetuning_method_comparison_report(
    *,
    samples: list[dict[str, Any]],
    zero_shot_predictions: list[str],
    pytorch_outputs: dict[str, list[str]] | None = None,
    trainer_outputs: dict[str, list[str]] | None = None,
    pytorch_comparison_metrics: dict[str, Any] | None = None,
    trainer_comparison_metrics: dict[str, Any] | None = None,
    pytorch_eval_metrics: dict[str, Any] | None = None,
    trainer_eval_metrics: dict[str, Any] | None = None,
    pytorch_model_dir: Path | str | None = None,
    trainer_model_dir: Path | str | None = None,
) -> dict[str, Any]:
    report_samples = [
        _caption_sample_row(sample, zero_shot_predictions[index])
        for index, sample in enumerate(samples)
    ]
    _add_comparison_predictions(
        report_samples=report_samples,
        outputs=pytorch_outputs,
        field_name="pytorch_fine_tuned",
    )
    _add_comparison_predictions(
        report_samples=report_samples,
        outputs=trainer_outputs,
        field_name="trainer_fine_tuned",
    )

    method_metrics: dict[str, Any] = {}
    if pytorch_comparison_metrics is not None:
        method_metrics.update(
            {
                "zero_shot": pytorch_comparison_metrics["zero_shot"],
                "pytorch_fine_tuned": pytorch_comparison_metrics["fine_tuned"],
                "pytorch_delta": pytorch_comparison_metrics["delta"],
            }
        )
    if trainer_comparison_metrics is not None:
        method_metrics.setdefault("zero_shot", trainer_comparison_metrics["zero_shot"])
        method_metrics.update(
            {
                "trainer_fine_tuned": trainer_comparison_metrics["fine_tuned"],
                "trainer_delta": trainer_comparison_metrics["delta"],
            }
        )

    method_eval_loss: dict[str, float | None] = {}
    method_models: dict[str, str] = {}
    if pytorch_eval_metrics is not None:
        method_eval_loss["pytorch"] = pytorch_eval_metrics.get("eval_loss")
        if pytorch_model_dir is not None:
            method_models["pytorch"] = str(pytorch_model_dir)
    if trainer_eval_metrics is not None:
        method_eval_loss["trainer"] = trainer_eval_metrics.get("eval_loss")
        if trainer_model_dir is not None:
            method_models["trainer"] = str(trainer_model_dir)

    return {
        "samples": report_samples,
        "metrics": method_metrics,
        "eval_loss": method_eval_loss,
        "models": method_models,
    }


def build_full_finetuning_visualization_sets(
    report: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    samples = report.get("samples", [])
    prediction_sets: dict[str, list[str]] = {"Zero-shot": [sample["zero_shot"] for sample in samples]}

    if any("pytorch_fine_tuned" in sample for sample in samples):
        prediction_sets["PyTorch fine-tuned"] = [
            sample["pytorch_fine_tuned"] for sample in samples if "pytorch_fine_tuned" in sample
        ]
    if any("trainer_fine_tuned" in sample for sample in samples):
        prediction_sets["Trainer fine-tuned"] = [
            sample["trainer_fine_tuned"] for sample in samples if "trainer_fine_tuned" in sample
        ]

    metrics = report.get("metrics", {})
    metric_sets: dict[str, dict[str, float]] = {}
    if "zero_shot" in metrics:
        metric_sets["Zero-shot"] = metrics["zero_shot"]
    if "pytorch_fine_tuned" in metrics:
        metric_sets["PyTorch fine-tuned"] = metrics["pytorch_fine_tuned"]
    if "trainer_fine_tuned" in metrics:
        metric_sets["Trainer fine-tuned"] = metrics["trainer_fine_tuned"]
    return prediction_sets, metric_sets


def evaluate_full_finetuning_caption_run(
    *,
    label: str,
    model: Any,
    processor: Any,
    sample_splits: dict[str, list[dict[str, Any]]],
    device: Any,
    prompt: str,
    max_new_tokens: int,
    batch_size: int,
    zero_shot_outputs: dict[str, list[str]],
    metric_samples: list[dict[str, Any]],
    comparison_samples: list[dict[str, Any]],
    output_dir: Path,
    model_dir: Path,
    config_payload: dict[str, Any],
    train_result_metrics: dict[str, Any] | None,
    eval_metrics: dict[str, Any] | None,
    log_history: list[dict[str, Any]],
    logger: Any | None = None,
) -> dict[str, Any]:
    from .caption_metrics import build_caption_before_after_report, safe_compute_caption_metric_comparison
    from .smolvlm_utils import generate_caption_splits
    from .training_utils import save_json

    outputs = generate_caption_splits(
        label=label,
        model=model,
        processor=processor,
        sample_splits=sample_splits,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        logger=logger,
    )
    comparison_metrics = safe_compute_caption_metric_comparison(
        zero_shot_predictions=zero_shot_outputs["test"],
        fine_tuned_predictions=outputs["test"],
        samples=metric_samples,
        logger=logger,
    )
    before_after_report = build_caption_before_after_report(
        samples=comparison_samples,
        zero_shot_predictions=zero_shot_outputs["comparison"],
        fine_tuned_predictions=outputs["comparison"],
        metrics=comparison_metrics,
    )
    save_json(output_dir / "before_after.json", before_after_report)
    save_json(
        output_dir / "train_eval_metrics.json",
        {
            "config": config_payload,
            "model_dir": str(model_dir),
            "train_result": train_result_metrics,
            "eval_metrics": eval_metrics,
            "log_history": log_history,
        },
    )
    return {
        "outputs": outputs,
        "comparison_metrics": comparison_metrics,
        "eval_metrics": eval_metrics,
    }


def _add_full_finetuning_sample_predictions(
    *,
    report_samples: list[dict[str, Any]],
    full_finetuning: dict[str, Any],
) -> None:
    full_samples = full_finetuning.get("samples")
    if not isinstance(full_samples, list):
        return
    predictions_by_identity = {
        _sample_identity(sample): sample.get("pytorch_fine_tuned")
        for sample in full_samples
        if isinstance(sample, dict) and sample.get("pytorch_fine_tuned") is not None
    }
    for sample in report_samples:
        prediction = predictions_by_identity.get(_sample_identity(sample))
        if prediction is not None:
            sample["full_finetuning_pytorch"] = prediction


def build_lora_method_comparison_report(
    *,
    samples: list[dict[str, Any]],
    zero_shot_predictions: list[str],
    zero_shot_scores: dict[str, Any],
    rank_results: list[dict[str, Any]],
    full_finetuning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report_samples = [
        _caption_sample_row(sample, zero_shot_predictions[index])
        for index, sample in enumerate(samples)
    ]
    method_metrics: dict[str, Any] = {"zero_shot": zero_shot_scores}
    method_eval_loss: dict[str, float | None] = {}
    method_models: dict[str, str] = {}

    if full_finetuning and full_finetuning.get("available"):
        full_metrics = full_finetuning.get("metrics", {})
        if isinstance(full_metrics, dict) and full_metrics.get("full_finetuned") is not None:
            method_metrics["full_finetuning_pytorch"] = full_metrics["full_finetuned"]
        if isinstance(full_metrics, dict) and full_metrics.get("full_ft_delta") is not None:
            method_metrics["full_finetuning_delta"] = full_metrics["full_ft_delta"]
        method_eval_loss["full_finetuning_pytorch"] = full_finetuning.get("eval_loss")
        if full_finetuning.get("model") is not None:
            method_models["full_finetuning_pytorch"] = str(full_finetuning["model"])
        _add_full_finetuning_sample_predictions(
            report_samples=report_samples,
            full_finetuning=full_finetuning,
        )

    for rank_result in rank_results:
        summary = rank_result["summary"]
        rank = int(summary["rank"])
        method_key = f"lora_rank_{rank}"
        before_after_report = rank_result.get("before_after_report", {})
        before_after_samples = before_after_report.get("samples", [])
        for index, sample_report in enumerate(before_after_samples):
            if index < len(report_samples) and isinstance(sample_report, dict):
                report_samples[index][method_key] = sample_report.get("fine_tuned")

        caption_scores = rank_result.get("caption_scores", summary.get("caption_scores", {}))
        if isinstance(caption_scores, dict) and caption_scores.get("available", True) is not False:
            if caption_scores.get("fine_tuned") is not None:
                method_metrics[f"{method_key}_fine_tuned"] = caption_scores["fine_tuned"]
            if caption_scores.get("delta") is not None:
                method_metrics[f"{method_key}_delta"] = caption_scores["delta"]
        method_eval_loss[method_key] = summary.get("eval_loss")
        if rank_result.get("rank_output_dir") is not None:
            method_models[method_key] = str(rank_result["rank_output_dir"])

    return {
        "samples": report_samples,
        "metrics": method_metrics,
        "eval_loss": method_eval_loss,
        "models": method_models,
    }


def _metric_columns(scores: Any) -> dict[str, float | None]:
    if not isinstance(scores, dict) or scores.get("available", True) is False:
        return {"bleu": None, "meteor": None, "cider_d": None}
    return {
        "bleu": None if "bleu" not in scores else float(scores["bleu"]),
        "meteor": None if "meteor" not in scores else float(scores["meteor"]),
        "cider_d": None if "cider_d" not in scores else float(scores["cider_d"]),
    }


def _comparison_row(
    *,
    method: str,
    rank: Any,
    trainable_params: Any,
    vs_full_finetuning_ratio: float | None,
    eval_loss: Any,
    scores: Any = None,
    note: Any = None,
) -> dict[str, Any]:
    return {
        "method": method,
        "rank": rank,
        "trainable_params": trainable_params,
        "vs_full_finetuning_ratio": vs_full_finetuning_ratio,
        "eval_loss": eval_loss,
        **_metric_columns(scores),
        "note": note,
    }


def build_method_comparison_rows(
    *,
    zero_shot_scores: dict[str, Any],
    full_finetuning: dict[str, Any],
    lora_rows: list[dict[str, Any]],
    logger: Any | None = None,
) -> list[dict[str, Any]]:
    rows = [
        _comparison_row(
            method="zero_shot",
            rank="-",
            trainable_params=0,
            vs_full_finetuning_ratio=0.0,
            eval_loss=None,
            scores=zero_shot_scores,
            note="no training",
        )
    ]
    if full_finetuning.get("available"):
        rows.append(
            _comparison_row(
                method="full_finetuning_pytorch",
                rank="-",
                trainable_params=full_finetuning.get("trainable_params"),
                vs_full_finetuning_ratio=1.0,
                eval_loss=full_finetuning.get("eval_loss"),
                scores=full_finetuning.get("metrics", {}).get("full_finetuned"),
                note="loaded from 1.5-VLM artifact",
            )
        )
    else:
        rows.append(
            _comparison_row(
                method="full_finetuning_missing",
                rank="-",
                trainable_params=None,
                vs_full_finetuning_ratio=1.0,
                eval_loss=None,
                note=full_finetuning.get("reason"),
            )
        )
        if logger is not None:
            logger.warning("Full fine-tuning comparison artifact is not available: %s", full_finetuning.get("path"))
    for row in lora_rows:
        caption_scores = row.get("caption_scores", {})
        fine_tuned_scores = caption_scores.get("fine_tuned") if isinstance(caption_scores, dict) else None
        rows.append(
            _comparison_row(
                method="lora",
                rank=row.get("rank"),
                trainable_params=row.get("trainable_params"),
                vs_full_finetuning_ratio=row.get("vs_full_finetuning_ratio"),
                eval_loss=row.get("eval_loss"),
                scores=fine_tuned_scores,
                note="adapter-only training",
            )
        )
    return rows


def log_method_comparison_rows(rows: list[dict[str, Any]], logger: Any) -> None:
    table_rows = [
        {
            "method": row["method"],
            "rank": row["rank"],
            "trainable_params": "n/a" if row["trainable_params"] is None else row["trainable_params"],
            "vs_full": "n/a" if row["vs_full_finetuning_ratio"] is None else f"{float(row['vs_full_finetuning_ratio']):.4%}",
            "eval_loss": "n/a" if row["eval_loss"] is None else f"{float(row['eval_loss']):.6f}",
            "bleu": "n/a" if row["bleu"] is None else f"{float(row['bleu']):.4f}",
            "meteor": "n/a" if row["meteor"] is None else f"{float(row['meteor']):.4f}",
            "cider_d": "n/a" if row["cider_d"] is None else f"{float(row['cider_d']):.4f}",
        }
        for row in rows
    ]
    logger.info(
        "Zero-shot / Full FT / LoRA comparison\n%s",
        build_fixed_width_table(
            rows=table_rows,
            columns=("method", "rank", "trainable_params", "vs_full", "eval_loss", "bleu", "meteor", "cider_d"),
        ),
    )


__all__ = [
    "build_fixed_width_table",
    "build_full_finetuning_method_comparison_report",
    "build_full_finetuning_visualization_sets",
    "build_lora_method_comparison_report",
    "build_method_comparison_rows",
    "evaluate_full_finetuning_caption_run",
    "load_full_finetuning_comparison",
    "log_before_after_report",
    "log_method_comparison_rows",
    "log_parameter_efficiency",
    "log_rank_summary",
]
