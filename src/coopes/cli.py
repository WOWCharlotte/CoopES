"""Command-line orchestration and experiment artifact management."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from io import StringIO
from pathlib import Path
from statistics import fmean

from .data import (
    DATASET_SPECS,
    DataConfig,
    DatasetSplits,
    LabeledQuestion,
    load_splits,
)
from .evaluation import (
    EvaluationResult,
    evaluate_trial,
    make_utility,
)
from .inference import (
    DecodeConfig,
    InferenceConfig,
    InferenceError,
    InferenceMetadata,
    InferenceStats,
    OllamaConfig,
    TransformersConfig,
    open_engine,
)
from .selection import (
    Method,
    SelectionConfig,
    SelectionPlan,
    SelectionTrial,
    build_selection_plan,
)

METHOD_ORDER: tuple[Method, ...] = (
    "zero_shot",
    "random",
    "singleton",
    "coopes_uniform",
    "coopes_weighted",
)


@dataclass(frozen=True)
class RunConfig:
    """Complete configuration for one experiment run."""

    data: DataConfig
    inference: InferenceConfig
    selection: SelectionConfig


@dataclass(frozen=True)
class TrialReport:
    """Evaluation output and timing for one frozen selection."""

    trial: SelectionTrial
    result: EvaluationResult
    elapsed_seconds: float


@dataclass(frozen=True)
class MethodSummary:
    """Mean metrics across all trials belonging to one method."""

    trial_count: int
    mean_accuracy: float
    mean_weighted_f1: float
    mean_parse_failure_rate: float


@dataclass(frozen=True)
class RunSummary:
    """Top-level run record persisted as ``summary.json``."""

    method_metrics: dict[Method, MethodSummary]
    inference_metadata: InferenceMetadata
    inference_stats: InferenceStats
    elapsed_seconds: float


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, value: object) -> None:
    content = json.dumps(
        _jsonable(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    _atomic_write_text(path, content + "\n")


def _load_json(path: Path) -> object:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"failed to read existing artifact {path}: {exc}"
        ) from exc


def _load_or_write_json(path: Path, value: object) -> None:
    expected = _jsonable(value)
    if path.exists():
        actual = _load_json(path)
        if actual != expected:
            raise ValueError(
                f"existing artifact {path} does not match the current run"
            )
        return
    _atomic_write_json(path, expected)


def _splits_payload(splits: DatasetSplits) -> object:
    return {
        "candidates": _jsonable(splits.candidates),
        "calibration": _jsonable(splits.calibration),
        "test": _jsonable(splits.test),
    }


def _config_payload(config: RunConfig) -> object:
    return _jsonable(config)


def _positive_int(
    table: Mapping[str, object],
    key: str,
    context: str,
) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer")
    return value


def _integer(
    table: Mapping[str, object],
    key: str,
    context: str,
) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}.{key} must be an integer")
    return value


def _positive_float(
    table: Mapping[str, object],
    key: str,
    context: str,
) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}.{key} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{context}.{key} must be a positive number")
    return result


def _text(
    table: Mapping[str, object],
    key: str,
    context: str,
) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _table(
    parent: Mapping[str, object],
    key: str,
    context: str,
) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}.{key} must be a table")
    return value


def _check_fields(
    table: Mapping[str, object],
    required: set[str],
    context: str,
) -> None:
    missing = sorted(required - set(table))
    unknown = sorted(set(table) - required)
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def _load_decode(
    inference: Mapping[str, object],
    context: str,
) -> DecodeConfig:
    decode = _table(inference, "decode", context)
    fields = {"seed", "max_new_tokens", "context_window"}
    _check_fields(decode, fields, f"{context}.decode")
    config = DecodeConfig(
        seed=_integer(decode, "seed", f"{context}.decode"),
        max_new_tokens=_positive_int(
            decode, "max_new_tokens", f"{context}.decode"
        ),
        context_window=_positive_int(
            decode, "context_window", f"{context}.decode"
        ),
    )
    if config.max_new_tokens >= config.context_window:
        raise ValueError(
            f"{context}.decode.max_new_tokens must be smaller than "
            "context_window"
        )
    return config


def load_config(config_path: Path) -> RunConfig:
    """Load and strictly validate a TOML experiment configuration."""

    try:
        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(
            f"failed to load config {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"config {config_path} must contain a TOML table")
    _check_fields(payload, {"data", "inference", "selection"}, "config")

    data = _table(payload, "data", "config")
    data_fields = {
        "dataset",
        "revision",
        "candidate_count",
        "calibration_count",
        "test_count",
        "split_seed",
    }
    _check_fields(data, data_fields, "config.data")
    dataset = _text(data, "dataset", "config.data")
    if dataset not in DATASET_SPECS:
        raise ValueError(
            "config.data.dataset must be one of "
            f"{', '.join(sorted(DATASET_SPECS))}"
        )
    data_config = DataConfig(
        dataset=dataset,
        revision=_text(data, "revision", "config.data"),
        candidate_count=_positive_int(data, "candidate_count", "config.data"),
        calibration_count=_positive_int(
            data, "calibration_count", "config.data"
        ),
        test_count=_positive_int(data, "test_count", "config.data"),
        split_seed=_integer(data, "split_seed", "config.data"),
    )

    inference = _table(payload, "inference", "config")
    backend = _text(inference, "backend", "config.inference")
    if backend == "ollama":
        inference_fields = {
            "backend",
            "base_url",
            "model_tag",
            "model_digest",
            "timeout_seconds",
            "max_attempts",
            "decode",
        }
        _check_fields(inference, inference_fields, "config.inference")
        inference_config: InferenceConfig = OllamaConfig(
            base_url=_text(inference, "base_url", "config.inference"),
            model_tag=_text(inference, "model_tag", "config.inference"),
            model_digest=_text(inference, "model_digest", "config.inference"),
            timeout_seconds=_positive_float(
                inference, "timeout_seconds", "config.inference"
            ),
            max_attempts=_positive_int(
                inference, "max_attempts", "config.inference"
            ),
            decode=_load_decode(inference, "config.inference"),
        )
    elif backend == "transformers":
        inference_fields = {
            "backend",
            "model_id",
            "revision",
            "device",
            "dtype",
            "batch_size",
            "decode",
        }
        _check_fields(inference, inference_fields, "config.inference")
        dtype = _text(inference, "dtype", "config.inference")
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(
                "config.inference.dtype must be float32, float16, or bfloat16"
            )
        inference_config = TransformersConfig(
            model_id=_text(inference, "model_id", "config.inference"),
            revision=_text(inference, "revision", "config.inference"),
            device=_text(inference, "device", "config.inference"),
            dtype=dtype,
            batch_size=_positive_int(
                inference, "batch_size", "config.inference"
            ),
            decode=_load_decode(inference, "config.inference"),
        )
    else:
        raise ValueError(
            "config.inference.backend must be 'ollama' or 'transformers'"
        )

    selection = _table(payload, "selection", "config")
    selection_fields = {
        "shot_count",
        "num_samples",
        "sample_seed",
        "random_seed",
        "random_trials",
        "epsilon",
    }
    _check_fields(selection, selection_fields, "config.selection")
    selection_config = SelectionConfig(
        shot_count=_positive_int(selection, "shot_count", "config.selection"),
        num_samples=_positive_int(
            selection, "num_samples", "config.selection"
        ),
        sample_seed=_integer(selection, "sample_seed", "config.selection"),
        random_seed=_integer(selection, "random_seed", "config.selection"),
        random_trials=_positive_int(
            selection, "random_trials", "config.selection"
        ),
        epsilon=_positive_float(selection, "epsilon", "config.selection"),
    )
    if selection_config.shot_count > data_config.candidate_count:
        raise ValueError(
            "config.selection.shot_count cannot exceed "
            "config.data.candidate_count"
        )
    return RunConfig(data_config, inference_config, selection_config)


def prepare_run(
    run_dir: Path,
    config: RunConfig,
    splits: DatasetSplits,
) -> None:
    """Create a run directory and validate resumable snapshots."""

    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "summary.json").exists():
        raise FileExistsError(
            f"run directory {run_dir} already contains summary.json"
        )
    _load_or_write_json(run_dir / "config.json", _config_payload(config))
    _load_or_write_json(run_dir / "splits.json", _splits_payload(splits))


def write_selection_artifacts(
    run_dir: Path,
    plan: SelectionPlan,
) -> None:
    """Write candidate scores and all frozen selection trials."""

    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["method", "candidate_id", "score", "sample_count"])
    for method in sorted(plan.scores):
        for score in plan.scores[method]:
            writer.writerow(
                [
                    method,
                    score.candidate_id,
                    "" if score.score is None else f"{score.score:.12g}",
                    score.sample_count,
                ]
            )
    _atomic_write_text(run_dir / "candidate_scores.csv", output.getvalue())
    _atomic_write_json(
        run_dir / "selections.json",
        {
            "trials": plan.trials,
            "missing_by_method": plan.missing_by_method,
        },
    )


def _mean_or_zero(values: Sequence[float]) -> float:
    return fmean(values) if values else 0.0


def write_evaluation_artifacts(
    run_dir: Path,
    trial_reports: Sequence[TrialReport],
    inference_metadata: InferenceMetadata,
    inference_stats: InferenceStats,
    elapsed_seconds: float,
) -> RunSummary:
    """Write predictions, trial metrics, and the run summary."""

    prediction_lines: list[str] = []
    metrics_output = StringIO()
    metrics_writer = csv.writer(
        metrics_output,
        lineterminator="\n",
    )
    metrics_writer.writerow(
        [
            "method",
            "trial_index",
            "question_count",
            "accuracy",
            "weighted_f1",
            "parse_failure_rate",
            "elapsed_seconds",
        ]
    )
    grouped: dict[Method, list[TrialReport]] = defaultdict(list)
    for report in trial_reports:
        grouped[report.trial.method].append(report)
        metrics = report.result.metrics
        metrics_writer.writerow(
            [
                report.trial.method,
                report.trial.trial_index,
                metrics.question_count,
                f"{metrics.accuracy:.12g}",
                f"{metrics.weighted_f1:.12g}",
                f"{metrics.parse_failure_rate:.12g}",
                f"{report.elapsed_seconds:.12g}",
            ]
        )
        for row in report.result.rows:
            prediction_lines.append(
                json.dumps(
                    {
                        "method": report.trial.method,
                        "trial_index": report.trial.trial_index,
                        "question_id": row.prediction.question_id,
                        "raw_text": row.prediction.raw_text,
                        "predicted_label": row.prediction.predicted_label,
                        "cache_hit": row.prediction.cache_hit,
                        "gold_label": row.gold_label,
                        "is_correct": row.is_correct,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

    method_metrics: dict[Method, MethodSummary] = {}
    for method in METHOD_ORDER:
        reports = grouped.get(method, [])
        if not reports:
            continue
        method_metrics[method] = MethodSummary(
            trial_count=len(reports),
            mean_accuracy=_mean_or_zero(
                [report.result.metrics.accuracy for report in reports]
            ),
            mean_weighted_f1=_mean_or_zero(
                [report.result.metrics.weighted_f1 for report in reports]
            ),
            mean_parse_failure_rate=_mean_or_zero(
                [
                    report.result.metrics.parse_failure_rate
                    for report in reports
                ]
            ),
        )
    summary = RunSummary(
        method_metrics=method_metrics,
        inference_metadata=inference_metadata,
        inference_stats=inference_stats,
        elapsed_seconds=elapsed_seconds,
    )

    predictions_content = "\n".join(prediction_lines) + (
        "\n" if prediction_lines else ""
    )
    _atomic_write_text(run_dir / "predictions.jsonl", predictions_content)
    _atomic_write_text(run_dir / "metrics.csv", metrics_output.getvalue())
    _atomic_write_json(run_dir / "summary.json", summary)
    return summary


def run_experiment(config: RunConfig, run_dir: Path) -> RunSummary:
    """Execute selection and frozen-selection evaluation end to end."""

    started_at = time.monotonic()
    splits = load_splits(config.data)
    prepare_run(run_dir, config, splits)
    candidates_by_id: dict[str, LabeledQuestion] = {
        item.question.question_id: item for item in splits.candidates
    }

    with open_engine(config.inference, run_dir / "cache.sqlite3") as engine:
        _load_or_write_json(
            run_dir / "inference_metadata.json",
            engine.metadata,
        )
        utility = make_utility(
            splits.candidates,
            splits.calibration,
            engine,
        )
        plan = build_selection_plan(
            tuple(candidates_by_id),
            utility,
            config.selection,
        )
        write_selection_artifacts(run_dir, plan)
        missing_methods = sorted(plan.missing_by_method)
        if missing_methods:
            details = ", ".join(
                f"{method}: {list(plan.missing_by_method[method])}"
                for method in missing_methods
            )
            raise ValueError(
                "candidate coverage is incomplete; test evaluation aborted "
                f"({details})"
            )

        trial_reports: list[TrialReport] = []
        for trial in plan.trials:
            examples = tuple(
                candidates_by_id[candidate_id]
                for candidate_id in trial.candidate_ids
            )
            trial_started_at = time.monotonic()
            result = evaluate_trial(examples, splits.test, engine)
            trial_reports.append(
                TrialReport(
                    trial=trial,
                    result=result,
                    elapsed_seconds=time.monotonic() - trial_started_at,
                )
            )
        stats = engine.snapshot_stats()
        metadata = engine.metadata
        elapsed_seconds = time.monotonic() - started_at
        return write_evaluation_artifacts(
            run_dir,
            trial_reports,
            metadata,
            stats,
            elapsed_seconds,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and run one experiment."""

    parser = argparse.ArgumentParser(prog="coopes")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        summary = run_experiment(config, args.run_dir)
    except (ValueError, OSError, InferenceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"summary: {args.run_dir / 'summary.json'}")
    method_count = len(summary.method_metrics)
    print(f"evaluated methods: {method_count}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
